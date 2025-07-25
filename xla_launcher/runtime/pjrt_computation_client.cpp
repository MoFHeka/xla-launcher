/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/runtime/pjrt_computation_client.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tsl/profiler/lib/traceme.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/literal.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/shape.h"
#include "xla_launcher/runtime/debug_macros.hpp"
#include "xla_launcher/runtime/env.hpp"
#include "xla_launcher/runtime/operation_manager.hpp"
#include "xla_launcher/runtime/pjrt_registry.hpp"
#include "xla_launcher/runtime/stablehlo_helper.hpp"
#include "xla_launcher/runtime/sys_util.hpp"
#include "xla_launcher/runtime/tensor_source.hpp"
#include "xla_launcher/runtime/xla_env_hash.hpp"
#include "xla_launcher/runtime/xla_util.hpp"

namespace xla_launcher {
namespace runtime {

//
// PjRT Client Implementation
//
//****************************************************************************

namespace {

// Builds a map from the device's global ordinal to its index in the `devices`
// array.
[[maybe_unused]] std::unordered_map<int, int> build_index_map(
  const std::vector<std::string>& devices) {
  std::unordered_map<int, int> device_index;
  for (int i = 0; i < devices.size(); ++i) {
    std::vector<std::string> device_spec = absl::StrSplit(devices[i], ':');
    XLA_CHECK_EQ(device_spec.size(), 2)
      << "Invalid device specification: " << devices[i];
    int global_ordinal = std::stoi(device_spec[1]);
    device_index[global_ordinal] = i;
  }
  return device_index;
}

// Builds the xla::Shape of the output xla::Literal on the host.
[[maybe_unused]] xla::Shape host_output_shape(xla::PjRtBuffer* buffer) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(
    buffer->element_type(), buffer->logical_dimensions().value());
  *shape.mutable_layout() = buffer->layout()->xla_layout();
  return xla::ShapeUtil::DeviceShapeToHostShape(shape);
}

[[maybe_unused]] hash_t hash_comp_env(
  xla::PjRtClient* client, std::vector<xla::PjRtDevice*>& ordered_devices) {
  hash_t hash = hash::HashXlaEnvVars();
  auto topology_desc = client->GetTopologyDescription();
  if (topology_desc.ok()) {
    // Some backends support a topology description which provides a better
    // view of the specific compilation environment.
    auto serialized = topology_desc.value()->Serialize();
    if (serialized.ok()) {
      return util::HashCombine(
        hash, util::DataHash(serialized->data(), serialized->length()));
    }
    // If serialization fails, fallthrough to the manual approach.
  }
  std::string platform_name(client->platform_name());
  std::string platform_version(client->platform_version());
  hash = util::HashCombine(hash, util::StringHash(platform_name.c_str()));
  // platform_version incorporates libtpu version and hardware type.
  hash = util::HashCombine(hash, util::StringHash(platform_version.c_str()));
  // Include global devices in the hash, ensuring order is consistent.
  for (auto& device : ordered_devices) {
    std::string device_str(device->ToString());
    hash = util::HashCombine(hash, util::StringHash(device_str.c_str()));
  }
  return hash;
}

}  // namespace

std::string PjRtComputationClient::PjRtDeviceToString(
  xla::PjRtDevice* const device) const {
  auto platform = device->client()->platform_name();
  int ordinal = global_ordinals_.at(device->id());
  std::string str = absl::StrFormat("%s:%d", platform, ordinal);
  return str;
}

std::vector<std::string> PjRtComputationClient::PjRtDevicesToString(
  absl::Span<xla::PjRtDevice* const> devices) const {
  std::vector<std::string> strs;
  strs.reserve(devices.size());

  for (auto* device : devices) {
    strs.push_back(PjRtDeviceToString(device));
  }

  return strs;
}

PjRtComputationClient::PjRtComputationClient(
  const ClientOptions& options,
  DeviceMemoryAllocatorFactory device_memory_allocator_factory,
  HostMemoryAllocatorFactory host_memory_allocator_factory) {
  client_ = InitializePjRt(
    options, device_memory_allocator_factory, host_memory_allocator_factory);

  // PjRtDevice IDs are not guaranteed to be dense, so we need to track
  // a device's global ordinal separately from its device ID. Order the
  // devices by increasing ID to assign global ordinals.
  std::vector<xla::PjRtDevice*> ordered_devices(client_->device_count());
  std::partial_sort_copy(
    client_->devices().begin(), client_->devices().end(),
    ordered_devices.begin(), ordered_devices.end(),
    [](auto& a, auto& b) { return a->id() < b->id(); });
  for (auto* device : ordered_devices) {
    global_ordinals_[device->id()] = global_ordinals_.size();
    std::string device_str = PjRtDeviceToString(device);
    string_to_device_.emplace(device_str, device);
  }
  comp_env_hash_ = hash_comp_env(client_.get(), ordered_devices);

  auto tracked_devices = GetLocalDevices();
  operation_manager_ = std::move(OperationManager(std::move(tracked_devices)));
}

PjRtComputationClient::~PjRtComputationClient() {
  WaitDeviceOps({});
  client_ = nullptr;
}

void PjRtComputationClient::PjRtData::Assign(const Data& data) {
  const PjRtData& pjrt_data = dynamic_cast<const PjRtData&>(data);
  if (&pjrt_data != this) {
    buffer = pjrt_data.buffer;
  }
}

ComputationClient::DataPtr PjRtComputationClient::CreateDataPlaceholder(
  std::string device, xla::Shape shape,
  std::optional<xla::OpSharding> sharding) {
  if (sharding.has_value()) {
    return std::make_shared<PjRtShardedData>(
      std::move(device), std::move(shape), std::move(*sharding));
  }

  return std::make_shared<PjRtData>(std::move(device), std::move(shape));
}

ComputationClient::DataPtr PjRtComputationClient::CreateData(
  std::string device, xla::Shape shape,
  std::shared_ptr<xla::PjRtBuffer> pjrt_buffer) {
  return std::make_shared<PjRtData>(
    std::move(device), std::move(shape), pjrt_buffer);
}

std::vector<ComputationClient::DataPtr> PjRtComputationClient::TransferToDevice(
  absl::Span<const std::shared_ptr<const TensorSource>> tensors) {
  // metrics::TimedSection timed(TransferToDeviceMetric());
  tsl::profiler::TraceMe activity(
    "PjRtComputationClient::TransferToDevice",
    tsl::profiler::TraceMeLevel::kInfo);
  std::vector<ComputationClient::DataPtr> datas;
  datas.reserve(tensors.size());
  // int64_t total_size = 0;
  for (auto& tensor : tensors) {
    xla::PjRtDevice* pjrt_device = StringToPjRtDevice(tensor->device());

    // total_size += xla::ShapeUtil::ByteSizeOf(tensor->shape());

    std::shared_ptr<xla::PjRtBuffer> buffer =
      std::move(client_
                  ->BufferFromHostBuffer(
                    tensor->data(), tensor->primitive_type(),
                    tensor->dimensions(), tensor->byte_strides(),
                    xla::PjRtClient::HostBufferSemantics::
                      kImmutableUntilTransferCompletes,
                    [tensor]() { /* frees tensor */ },
                    *pjrt_device->default_memory_space(),
                    /*device_layout=*/nullptr)
                  .value());

    ComputationClient::DataPtr data =
      std::make_shared<PjRtData>(tensor->device(), tensor->shape(), buffer);
    datas.push_back(data);
  }
  // OutboundDataMetric()->AddSample(total_size);
  // CreateDataHandlesCounter()->AddValue(datas.size());

  return datas;
}

ComputationClient::DataPtr PjRtComputationClient::CopyToDevice(
  ComputationClient::DataPtr data, std::string dst) {
  tsl::profiler::TraceMe activity(
    "PjRtComputationClient::CopyToDevice", tsl::profiler::TraceMeLevel::kInfo);
  const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(data.get());
  XLA_CHECK(pjrt_data->HasValue()) << "Can't copy invalid device data.";

  xla::PjRtDevice* dst_device = StringToPjRtDevice(dst);
  XLA_CHECK(dst_device->IsAddressable()) << dst << "is not addressable.";

  // Returns error if the buffer is already on `dst_device`.
  absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> status_or =
    pjrt_data->buffer->CopyToMemorySpace(*dst_device->default_memory_space());
  if (!status_or.ok()) {
    return data;
  }
  return std::make_shared<PjRtData>(
    dst, pjrt_data->shape(), std::move(status_or.value()));
}

std::uintptr_t PjRtComputationClient::UnsafeBufferPointer(
  const ComputationClient::DataPtr handle) {
  std::shared_ptr<PjRtData> pjrt_data =
    std::dynamic_pointer_cast<PjRtData>(handle);
  XLA_CHECK(pjrt_data) << "handle must be PjRtData, got " << handle->ToString();
  XLA_CHECK(pjrt_data->buffer != nullptr)
    << "PjRt buffer is null in " << __FUNCTION__;
  absl::StatusOr<std::uintptr_t> ptr =
    client_->UnsafeBufferPointer(pjrt_data->buffer.get());
  XLA_CHECK(ptr.ok());
  return ptr.value();
}

std::shared_ptr<xla::PjRtBuffer> PjRtComputationClient::GetPjRtBuffer(
  const ComputationClient::DataPtr handle) const {
  std::shared_ptr<PjRtData> pjrt_data =
    std::dynamic_pointer_cast<PjRtData>(handle);

  XLA_CHECK(pjrt_data) << "handle must be PjRtData, got " << handle->ToString();
  std::shared_ptr<xla::PjRtBuffer> pjrt_buffer = pjrt_data->buffer;
  if (pjrt_buffer != nullptr) {
    return pjrt_buffer;
  } else {
    VLOG(3) << "The pjrt buffer is null so we need to wait for device ops "
               "to finish.";
    WaitDeviceOps({});
    return std::dynamic_pointer_cast<PjRtData>(handle)->buffer;
  }
}

std::vector<xla::Literal> PjRtComputationClient::TransferFromDevice(
  absl::Span<const ComputationClient::DataPtr> handles) {
  // metrics::TimedSection timed(TransferFromDeviceMetric());
  tsl::profiler::TraceMe activity(
    "PjRtComputationClient::TransferFromDevice",
    tsl::profiler::TraceMeLevel::kInfo);
  std::vector<xla::PjRtFuture<>> futures;
  futures.reserve(handles.size());
  std::vector<xla::Literal> literals;
  literals.reserve(handles.size());
  int64_t total_size = 0;
  for (auto handle : handles) {
    // Not support replication device for now
    std::shared_ptr<PjRtData> pjrt_data =
      std::dynamic_pointer_cast<PjRtData>(handle);
    XLA_CHECK(pjrt_data) << "PjRt_data is null in " << __FUNCTION__;
    XLA_CHECK(pjrt_data->buffer != nullptr)
      << "PjRt buffer is null in " << __FUNCTION__;

    xla::Literal& literal =
      literals.emplace_back(host_output_shape(pjrt_data->buffer.get()));
    futures.push_back(pjrt_data->buffer->ToLiteral(&literal));

    total_size += literal.size_bytes();
  }
  for (auto& future : futures) {
    absl::Status status = future.Await();
    XLA_CHECK_OK(status) << "Failed to await future from buffer to literal in"
                         << __FUNCTION__;
  }
  // InboundDataMetric()->AddSample(total_size);

  return literals;
}

std::vector<ComputationClient::ComputationPtr> PjRtComputationClient::Compile(
  std::vector<CompileInstance> instances) {
  // auto metrics_fn = CompileMetric;
  // metrics::TimedSection timed(metrics_fn());
  tsl::profiler::TraceMe activity(
    "PjRtComputationClient::Compile", tsl::profiler::TraceMeLevel::kInfo);
  std::vector<ComputationClient::ComputationPtr> computations;
  static bool enable_cm_in_mp =
    sys_util::GetEnvBool("ENABLE_COLLECTIVE_MATMUL_IN_MP", false);

  for (auto& instance : instances) {
    xla::CompileOptions compile_options;
    if (enable_cm_in_mp) {
      compile_options.executable_build_options.set_use_spmd_partitioning(true);
      compile_options.env_option_overrides.push_back(
        {"xla_tpu_decompose_all_gather_einsum", true});
      compile_options.env_option_overrides.push_back(
        {"xla_tpu_decompose_einsum_reduce_scatter", true});
    }
    if (instance.is_sharded) {
      // TODO(yeounoh) multi-host, multi-slice configurations
      compile_options.executable_build_options.set_use_spmd_partitioning(true);

      // We can override the compiler's default behavior to replicate the
      // outputs. Setting this to true would wrapping the sharded outputs in
      // PjRtShardedData.
      compile_options.executable_build_options
        .set_allow_spmd_sharding_propagation_to_output(
          {instance.allow_spmd_sharding_propagation_to_output});

      int num_partitions = client_->device_count();
      compile_options.executable_build_options.set_num_partitions(
        num_partitions);
      compile_options.executable_build_options.set_num_replicas(1);
      compile_options.parameter_is_tupled_arguments =
        instance.parameter_is_tupled_arguments;
      compile_options.executable_build_options.set_use_auto_spmd_partitioning(
        instance.use_auto_spmd_partitioning);
      VLOG(3) << "Auto SPMD partitioning "
              << (instance.use_auto_spmd_partitioning ? "enabled!"
                                                      : "disabled.");
      if (!instance.auto_spmd_mesh_shape.empty()) {
        compile_options.executable_build_options
          .set_auto_spmd_partitioning_mesh_shape(instance.auto_spmd_mesh_shape);
        VLOG(3) << "auto_spmd_partitioning_mesh_shape="
                << absl::StrJoin(
                     compile_options.executable_build_options
                       .auto_spmd_partitioning_mesh_shape(),
                     ",");
      }
      if (!instance.auto_spmd_mesh_ids.empty()) {
        compile_options.executable_build_options
          .set_auto_spmd_partitioning_mesh_ids(instance.auto_spmd_mesh_ids);
        VLOG(3) << "auto_spmd_partitioning_mesh_ids="
                << absl::StrJoin(
                     compile_options.executable_build_options
                       .auto_spmd_partitioning_mesh_ids(),
                     ",");
      }

      // TODO(244391366) verify this is correct for the collectives ops
      xla::DeviceAssignment device_assignment(1, client_->device_count());
      // DeviceAssignment values must be the PjRtDevice ID, so we need to
      // unwind the global ordinal mapping.
      for (const auto& [device_id, global_ordinal] : global_ordinals_) {
        device_assignment(0, global_ordinal) = device_id;
      }
      compile_options.executable_build_options.set_device_assignment(
        device_assignment);
    } else {
      // TODO(wcromar): set compile_options.argument_layouts, enable strict
      // shapes
      compile_options.executable_build_options.set_num_partitions(1);
      compile_options.executable_build_options.set_num_replicas(
        client_->device_count());
      compile_options.parameter_is_tupled_arguments =
        instance.parameter_is_tupled_arguments;

      xla::DeviceAssignment device_assignment(client_->device_count(), 1);
      // DeviceAssignment values must be the PjRtDevice ID, so we need to
      // unwind the global ordinal mapping.
      for (const auto& [device_id, global_ordinal] : global_ordinals_) {
        device_assignment(global_ordinal, 0) = device_id;
      }
      compile_options.executable_build_options.set_device_assignment(
        device_assignment);
    }

    // Compile the computation to an executible. For better user experience, if
    // the XLA compiler fails for any reason, we raise a Python exception.
    std::unique_ptr<xla::PjRtLoadedExecutable> executable;
    if (sys_util::GetEnvBool("XLA_STABLEHLO_COMPILE", false)) {
      // Convert HLO to StableHLO for PjRt client compilation.
      mlir::MLIRContext context;
      mlir::ModuleOp mlir_module =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
      util::ConvertHloToStableHlo(
        instance.computation.mutable_proto(), &mlir_module);
      executable = util::RaiseValueErrorOnFailure(
        [&] { return client_->CompileAndLoad(mlir_module, compile_options); });
      // StableHloCompileCounter()->AddValue(1);
    } else {
      executable = util::RaiseValueErrorOnFailure([&] {
        return client_->CompileAndLoad(instance.computation, compile_options);
      });
    }

    auto memory_stats_status_or = executable->GetCompiledMemoryStats();
    if (memory_stats_status_or.ok()) {
      xla::CompiledMemoryStats memory_stats = memory_stats_status_or.value();
      VLOG(3) << "memory usage detail = " << memory_stats.DebugString();
    } else {
      VLOG(3) << "memory usage is not availiable";
    }

    const auto& hlo_modules = ConsumeValue(executable->GetHloModules());
    [[maybe_unused]] xla::HloComputation* hlo_computation =
      hlo_modules[0]->entry_computation();
    std::shared_ptr<PjRtComputation> pjrt_computation =
      std::make_shared<PjRtComputation>(
        std::move(xla::XlaComputation(hlo_modules[0]->ToProto())),
        instance.devices, std::move(executable));

    computations.push_back(pjrt_computation);

    // CreateCompileHandlesCounter()->AddValue(1);
  }

  return computations;
}

std::string PjRtComputationClient::SerializeComputation(
  const ComputationPtr computation) const {
  const PjRtComputation& pjrt_computation =
    dynamic_cast<const PjRtComputation&>(*computation);

  return ConsumeValue(pjrt_computation.executable->SerializeExecutable());
}

ComputationClient::ComputationPtr PjRtComputationClient::DeserializeComputation(
  const std::string& serialized) const {
  absl::StatusOr<std::unique_ptr<xla::PjRtExecutable>> executable_or =
    client_->DeserializeExecutable(serialized, std::nullopt);
  if (!executable_or.ok()) {
    LOG(WARNING) << "Failed to deserialize executable: "
                 << executable_or.status();
    return nullptr;
  }
  std::unique_ptr<xla::PjRtExecutable> executable =
    std::move(executable_or.value());
  std::unique_ptr<xla::PjRtLoadedExecutable> loaded_executable =
    client_->Load(std::move(executable), xla::LoadOptions()).value();

  auto hlo_modules = loaded_executable->GetHloModules();
  if (!hlo_modules.ok()) {
    LOG(WARNING)
      << "Failed to retrieve HLO modules from deserialized executable";
    return nullptr;
  }
  XLA_CHECK(hlo_modules->size() == 1)
    << "Only a single module is supported for persistent computation "
       "caching. Please unset the XLA_PERSISTENT_CACHE_PATH "
       "variable to disable persistent caching.";
  xla::XlaComputation computation((*hlo_modules)[0]->ToProto());

  std::vector<std::string> devices = {GetDefaultDevice()};
  return std::make_shared<PjRtComputation>(
    std::move(computation), devices, std::move(loaded_executable));
}

hash_t PjRtComputationClient::HashCompilationEnv() {
  // TODO(jonbolin): Incorporate CompileOptions into the hash. These are
  // deterministically generated at the moment, so they don't need to be
  // included. It will require a small refactor, so punting on this for now.
  return comp_env_hash_;
}

std::vector<ComputationClient::DataPtr>
PjRtComputationClient::ExecuteComputation(
  const Computation& computation,
  absl::Span<const ComputationClient::DataPtr> arguments,
  const std::string& device, const ExecuteComputationOptions& options) {
  // Shared ownership of the timed section ensures that it will only get logged
  // once both `ExecuteComputation` and the async work in `ExecuteSharded` are
  // complete; a copy is held from the lambda that releases it when done.
  // auto metrics_fn = ExecuteMetric;
  // auto timed = std::make_shared<metrics::TimedSection>(metrics_fn());
  tsl::profiler::TraceMe activity(
    "PjRtComputationClient::ExecuteComputation",
    tsl::profiler::TraceMeLevel::kInfo);
  VLOG(1) << "Executing PjRt computation on " << device;
  const PjRtComputation& pjrt_computation =
    dynamic_cast<const PjRtComputation&>(computation);

  xla::PjRtDevice* pjrt_device = StringToPjRtDevice(device);
  XLA_CHECK(pjrt_device->IsAddressable()) << pjrt_device->DebugString();

  std::vector<xla::PjRtBuffer*> buffers;
  buffers.reserve(arguments.size());
  for (auto& argument : arguments) {
    const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(argument.get());

    XLA_CHECK(pjrt_device == pjrt_data->buffer->device())
      << "The device currently being used : " << pjrt_device->DebugString()
      << " is different from the device where the buffer resides: "
      << pjrt_data->buffer->device()->DebugString();
    buffers.push_back(pjrt_data->buffer.get());
  }

  xla::ExecuteOptions execute_options;
  execute_options.untuple_result = options.explode_tuple;
  execute_options.strict_shape_checking = false;

  // Required as of cl/518733871
  execute_options.use_major_to_minor_data_layout_for_callbacks = true;

  VLOG(5) << "ExecuteComputation acquiring PJRT device lock for " << device;
  auto op_tracker = operation_manager_.StartOperation(device);
  VLOG(5) << "ExecuteComputation acquiring PJRT device lock for " << device
          << " Done";

  std::optional<xla::PjRtFuture<>> returned_future;
  std::vector<std::unique_ptr<xla::PjRtBuffer>> results =
    pjrt_computation.executable
      ->ExecuteSharded(buffers, pjrt_device, execute_options, returned_future)
      .value();

  returned_future->OnReady(
    std::move([/*timed,*/ op_tracker =
                 std::move(op_tracker)](absl::Status unused) mutable {
      // timed.reset();
      VLOG(3) << "ExecuteComputation returned_future->OnReady finished";
    }));

  std::vector<DataPtr> datas;
  datas.reserve(results.size());
  for (auto& result : results) {
    std::unique_ptr<xla::PjRtBuffer> buffer = std::move(result);

    std::shared_ptr<PjRtData> data =
      std::make_shared<PjRtData>(device, std::move(buffer));

    datas.push_back(data);
  }
  // CreateDataHandlesCounter()->AddValue(datas.size());

  VLOG(1) << "Returning " << datas.size() << " results";
  return datas;
}

size_t PjRtComputationClient::GetNumLocalDevices() const {
  return client_->addressable_device_count();
}

size_t PjRtComputationClient::GetNumDevices() const {
  return client_->device_count();
}

std::string PjRtComputationClient::GetDefaultDevice() const {
  return PjRtDeviceToString(client_->addressable_devices()[0]);
}

DeviceType PjRtComputationClient::GetDeviceType() const {
  return DeviceType(client_->platform_name());
}

std::vector<std::string> PjRtComputationClient::GetLocalDevices() const {
  return PjRtDevicesToString(client_->addressable_devices());
}

std::vector<std::string> PjRtComputationClient::GetAllDevices() const {
  return PjRtDevicesToString(client_->devices());
}

int PjRtComputationClient::GetNumProcesses() const {
  int max_process_index = client_->process_index();
  for (auto* device : client_->devices()) {
    max_process_index = std::max(max_process_index, device->process_index());
  }

  return max_process_index + 1;
}

std::string PjRtComputationClient::GetDeviceKind(const std::string& device) {
  return std::string(StringToPjRtDevice(device)->device_kind());
}

std::intptr_t PjRtComputationClient::GetCudaStreamForDevice(
  int local_device_id) const {
  absl::StatusOr<xla::PjRtDevice*> pjrt_device =
    client_->LookupAddressableDevice(xla::PjRtLocalDeviceId(local_device_id));
  XLA_CHECK(pjrt_device.ok()) << "Failed to get a PjRt device.";
  absl::StatusOr<std::intptr_t> stream =
    pjrt_device.value()->GetStreamForExternalReadyEvents();
  XLA_CHECK(stream.ok()) << "Failed to get a stream.";
  return stream.value();
}

const std::unordered_map<std::string, ComputationClient::DeviceAttribute>
PjRtComputationClient::GetDeviceAttributes(const std::string& device) {
  auto attrs = PjRtComputationClient::StringToPjRtDevice(device)->Attributes();
  return std::unordered_map<std::string, ComputationClient::DeviceAttribute>(
    attrs.begin(), attrs.end());
}

xla::PjRtDevice* PjRtComputationClient::StringToPjRtDevice(
  const std::string& device) {
  XLA_CHECK(string_to_device_.find(device) != string_to_device_.end())
    << "Unknown device " << device;
  xla::PjRtDevice* pjrt_device = string_to_device_[device];
  return pjrt_device;
}

void PjRtComputationClient::WaitDeviceOps(
  absl::Span<const std::string> devices) const {
  VLOG(3) << "Waiting for " << absl::StrJoin(devices, ", ");
  operation_manager_.WaitForDevices(
    devices.empty() ? GetLocalDevices() : devices);
}

// std::map<std::string, Metric> PjRtComputationClient::GetMetrics() const {
//   // TODO(jonbolin): Add any PJRt-client-specific metrics here
//   return {};
// }

PjRtComputationClient::MemoryInfo PjRtComputationClient::GetMemoryInfo(
  const std::string& device) {
  xla::PjRtDevice* pjrt_device =
    PjRtComputationClient::StringToPjRtDevice(device);
  tsl::AllocatorStats stats = pjrt_device->GetAllocatorStats().value();

  return {
    stats.bytes_in_use,
    *stats.bytes_limit,
    stats.peak_bytes_in_use,
  };
}

void PjRtComputationClient::RegisterCustomCall(
  const std::string& fn_name, void* function_ptr, const std::string& platform) {
  if (platform != xla::CudaName()) {
    XLA_ERROR() << "Custom call targets can only be registered for "
                   "PJRT CUDA runtime.";
    return;
  }

  auto* c_api_client = dynamic_cast<xla::PjRtCApiClient*>(client_.get());
  if (!c_api_client) {
    XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(fn_name, function_ptr, platform);
    return;
  }
  const PJRT_Api* pjrt_api = c_api_client->pjrt_c_api();

  // See openxla reference:
  // https://github.com/openxla/xla/blob/b604c8d87df842002a7a8de79a434026329fbcb2/xla/pjrt/c/pjrt_c_api_gpu_test.cc#L414
  const PJRT_Extension_Base* next =
    reinterpret_cast<const PJRT_Extension_Base*>(pjrt_api->extension_start);
  while (next != nullptr
         && next->type
              != PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call) {
    next = next->next;
  }
  XLA_CHECK(next) << "Custom call extension not found";
  PJRT_Gpu_Register_Custom_Call_Args args;
  args.struct_size = PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE;
  args.function_name = fn_name.c_str();
  args.function_name_size = fn_name.size();
  args.api_version = 0;
  args.handler_execute = function_ptr;
  PJRT_Error* error =
    reinterpret_cast<const PJRT_Gpu_Custom_Call*>(next)->custom_call(&args);
  if (error) {
    XLA_ERROR() << error->status;
  }
}

void PjRtComputationClient::OnReadyCallback(
  ComputationClient::DataPtr data, const std::function<void()>& callback) {
  std::shared_ptr<xla::PjRtBuffer> buffer;
  if (auto pjrt_data = std::dynamic_pointer_cast<PjRtData>(data)) {
    buffer = pjrt_data->buffer;
  } else {
    XLA_ERROR() << "received invalid data pointer";
  }
  XLA_CHECK(buffer) << "received placeholder data as argument";
  buffer->GetReadyFuture().OnReady(
    [callback](absl::Status unused) { callback(); });
}

}  // namespace runtime
}  // namespace xla_launcher
