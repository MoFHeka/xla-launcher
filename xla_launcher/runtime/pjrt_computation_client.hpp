/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_RUNTIME_PJRT_COMPUTATION_CLIENT_HPP
#define XLA_LAUNCHER_RUNTIME_PJRT_COMPUTATION_CLIENT_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/shape_util.h"
#include "xla_launcher/device.hpp"
#include "xla_launcher/runtime/computation_client.hpp"
#include "xla_launcher/runtime/debug_macros.hpp"
#include "xla_launcher/runtime/operation_manager.hpp"
#include "xla_launcher/runtime/tensor_source.hpp"

namespace xla_launcher {
namespace runtime {

class PjRtComputationClient : public ComputationClient {
 public:
  explicit PjRtComputationClient(
    const ClientOptions& options = {},
    DeviceMemoryAllocatorFactory device_memory_allocator_factory = nullptr,
    HostMemoryAllocatorFactory host_memory_allocator_factory = nullptr);
  ~PjRtComputationClient();

  // Creates a Data object with no actual device handle in it. The device handle
  // will be populated in an asynchrounous fusion.
  DataPtr CreateDataPlaceholder(
    std::string device, xla::Shape shape,
    std::optional<xla::OpSharding> sharding) override;

  static DataPtr CreateData(
    std::string device, xla::Shape shape,
    std::shared_ptr<xla::PjRtBuffer> pjrt_buffer);

  std::string PjRtDeviceToString(xla::PjRtDevice* const device) const override;

  std::vector<std::string> PjRtDevicesToString(
    absl::Span<xla::PjRtDevice* const> devices) const;

  // Transfers local tensor values to the GPU devices and fetches the handles.
  std::vector<DataPtr> TransferToDevice(
    absl::Span<const std::shared_ptr<const TensorSource>> tensors) override;

  // Copies `data->buffer` to `dst` device buffer.
  DataPtr CopyToDevice(DataPtr data, std::string dst) override;

  // Creates a new Data object with the same device handle as `data` but with
  // a different shape.
  DataPtr Reshape(DataPtr data, xla::Shape shape);

  // Reads the tensor literal values stored at TPU server sites, behind the
  // supplied handles.
  // Note: `TransferFromDevice` call will block until the `DataPtrs` are ready
  // if they were created by `TransferToDevice` or `Execute*`. Calling this from
  // python while holding the GIL can cause deadlocks!
  std::vector<xla::Literal> TransferFromDevice(
    absl::Span<const DataPtr> handles) override;

  std::uintptr_t UnsafeBufferPointer(const DataPtr handle) override;

  std::shared_ptr<xla::PjRtBuffer> GetPjRtBuffer(
    const DataPtr handle) const override;

  // Compiles a set of computations.
  std::vector<ComputationPtr> Compile(
    std::vector<CompileInstance> instances) override;

  // Serialize a computation to a string.
  std::string SerializeComputation(
    const ComputationPtr computation) const override;

  // Deserialize a string resulting from SerializeComputation back to a
  // Computation. If the deserialization fails, nullptr is returned.
  ComputationPtr DeserializeComputation(
    const std::string& serialized) const override;

  // Returns a hash of the current compilation environment.
  hash_t HashCompilationEnv() override;

  // Executes computation with arguments and returns the result.
  // The passed device must match the common device of the arguments Data.
  // If options.explode_tuple is true, the output tuple will be decomposed into
  // its single elements.
  std::vector<DataPtr> ExecuteComputation(
    const Computation& computation, absl::Span<const DataPtr> arguments,
    const std::string& device,
    const ExecuteComputationOptions& options =
      ExecuteComputationOptions{}) override;

  std::string GetDefaultDevice() const override;

  DeviceType GetDeviceType() const override;

  std::string GetDeviceKind(const std::string& device) override;

  xla::PjRtPlatformId GetPlatformID() const override {
    return client_->platform_id();
  }

  absl::StatusOr<xla::PjRtDevice*> LookupAddressableDevice(
    int local_device_id) const override {
    return client_->LookupAddressableDevice(
      xla::PjRtLocalDeviceId(local_device_id));
  }

  std::intptr_t GetCudaStreamForDevice(int local_device_id) const override;

  size_t GetNumLocalDevices() const override;

  size_t GetNumDevices() const override;

  std::vector<std::string> GetLocalDevices() const override;

  std::vector<std::string> GetAllDevices() const override;

  int GetProcessIndex() const override { return client_->process_index(); }

  int GetNumProcesses() const override;

  using DeviceAttribute =
    std::variant<std::string, bool, int64_t, std::vector<int64_t>, float>;

  const std::unordered_map<std::string, DeviceAttribute> GetDeviceAttributes(
    const std::string& device) override;

  MemoryInfo GetMemoryInfo(const std::string& device) override;

  // Block until pass in devices' async operation are finished. If empty, all
  // the local devices will be waited for.
  void WaitDeviceOps(absl::Span<const std::string> devices = {}) const override;

  void RegisterCustomCall(
    const std::string& fn_name, void* function_ptr,
    const std::string& platform) override;

  // Installs a callback to be called when the buffer backing `data` is ready.
  void OnReadyCallback(
    DataPtr data, const std::function<void()>& callback) override;

 private:
  // Easy to test.
  friend class PjRtComputationClientTest;

  std::unique_ptr<xla::PjRtClient> client_;
  std::unordered_map<int, int> global_ordinals_;
  std::unordered_map<std::string, xla::PjRtDevice*> string_to_device_;
  mutable OperationManager operation_manager_;
  tsl::thread::ThreadPool pool_ = tsl::thread::ThreadPool(
    tsl::Env::Default(), "pjrt", std::thread::hardware_concurrency());
  hash_t comp_env_hash_;

  xla::PjRtDevice* StringToPjRtDevice(const std::string& device);

  struct PjRtData : public Data {
    PjRtData(std::string device, xla::Shape device_shape)
      : Data(std::move(device), std::move(device_shape)) {}

    PjRtData(
      std::string device, xla::Shape device_shape,
      std::shared_ptr<xla::PjRtBuffer> buffer)
      : Data(std::move(device), std::move(device_shape)), buffer(buffer) {}

    PjRtData(std::string device, std::shared_ptr<xla::PjRtBuffer> buffer)
      : Data(
        std::move(device), xla::Shape(
                             buffer->element_type(), buffer->dimensions(),
                             buffer->is_dynamic_dimension())),
        buffer(buffer) {}

    Handle GetHandle() override {
      // If the data is a placeholder, use the address of this object as the
      // handle.
      if (buffer == nullptr) {
        return reinterpret_cast<std::uintptr_t>(this);
      }

      XLA_CHECK(!buffer->IsDeleted())
        << "buffer with shape " << shape().ToString() << " on device "
        << device() << " is deleted";
      return reinterpret_cast<std::uintptr_t>(buffer.get());
    };
    void Assign(const Data& data) override;
    bool HasValue() const override {
      return buffer != nullptr && !buffer->IsDeleted();
    };

    bool HasSharding() const override { return false; }

    xla::OpSharding GetSharding() const override {
      XLA_CHECK(false) << "GetSharding should not be called on PjRtData, check "
                          "HasSharding first";
      return xla::OpSharding();
    }

    std::string ToString() const override {
      std::stringstream ss;
      ss << "XLAData: \n";
      ss << "  Data Device: " << device() << "\n";
      ss << "  Data Shape: " << shape().ToString() << "\n";
      ss << "  Data Handle: ";
      if (HasValue()) {
        ss << reinterpret_cast<std::uintptr_t>(buffer.get()) << "\n";
      } else {
        ss << (buffer == nullptr ? "None" : "Deleted") << "\n";
      }
      return ss.str();
    }

    std::shared_ptr<xla::PjRtBuffer> buffer;
  };

  struct PjRtShardedData : public Data {
    PjRtShardedData(std::string device, xla::Shape shape) = delete;

    PjRtShardedData(
      std::string device, xla::Shape shape, xla::OpSharding sharding)
      : Data(std::move(device), std::move(shape)), sharding(sharding) {}

    PjRtShardedData(
      std::string device, xla::Shape shape,
      std::vector<std::shared_ptr<PjRtData>> shards, xla::OpSharding sharding)
      : Data(std::move(device), std::move(shape)),
        shards(shards),
        sharding(sharding) {}

    Handle GetHandle() override {
      // Always returns `Handle` of the first shard.
      return shards[0]->GetHandle();
    }

    void Assign(const Data& data) override {
      const PjRtShardedData& pjrt_sharded_data =
        dynamic_cast<const PjRtShardedData&>(data);
      if (&pjrt_sharded_data != this) {
        shards = std::move(pjrt_sharded_data.shards);
      }
    }

    bool HasValue() const override {
      if (shards.empty()) {
        return false;
      }

      for (auto& shard : shards) {
        if (!shard->HasValue()) {
          return false;
        }
      }
      return true;
    }

    std::string ToString() const override {
      std::stringstream ss;
      ss << "XLAShardedData: \n";
      ss << "  Data Device: " << device() << "\n";
      ss << "  Data Shape: " << shape().ToString() << "\n";
      ss << "  OpSharding: "
         << xla::HloSharding::FromProto(sharding)->ToString() << "\n";
      ss << "  NumShards: " << shards.size() << "\n";
      return ss.str();
    }

    bool HasSharding() const override { return true; }

    xla::OpSharding GetSharding() const override { return sharding; }

    std::vector<std::shared_ptr<PjRtData>> shards;
    xla::OpSharding sharding;
  };

  struct PjRtComputation : public Computation {
    PjRtComputation(
      xla::XlaComputation computation, std::vector<std::string> devices,
      std::unique_ptr<xla::PjRtLoadedExecutable> executable)
      : Computation(std::move(computation), std::move(devices)),
        executable(std::move(executable)) {
      output_shardings_ = this->executable->GetOutputShardings();
    }

    const std::string get_memory_info() const override {
      auto memory_stats_status_or = executable->GetCompiledMemoryStats();
      if (memory_stats_status_or.ok()) {
        return memory_stats_status_or.value().DebugString();
      } else {
        return "memory usage is not available";
      }
    }

    std::unique_ptr<xla::PjRtLoadedExecutable> executable;
    std::optional<std::vector<xla::OpSharding>> output_shardings_;
  };
};

}  // namespace runtime
}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_PJRT_COMPUTATION_CLIENT_HPP
