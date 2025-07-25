/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/runtime/pjrt_registry.hpp"

#include <csignal>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/log/initialize.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/tsl/framework/device_id.h"
#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020
#include "xla/stream_executor/gpu/gpu_cudamallocasync_allocator.h"
#endif

#include "xla_launcher/device.hpp"
#include "xla_launcher/runtime/debug_macros.hpp"
#include "xla_launcher/runtime/env.hpp"
#include "xla_launcher/runtime/profiler.hpp"
#include "xla_launcher/runtime/sys_util.hpp"

namespace xla_launcher {
namespace runtime {

namespace {

// Placeholder plugin for example only. Does not implement multiprocessing or
// configuration. Very likely will not work from Python code.
class LibraryPlugin : public PjRtPlugin {
 public:
  std::string library_path() const override {
    return sys_util::GetEnvString("PJRT_LIBRARY_PATH", "");
  }

  const std::unordered_map<std::string, xla::PjRtValueType>
  client_create_options() const override {
    return {};
  }

  bool requires_xla_coordinator() const override { return false; }
};

std::unordered_map<std::string, std::shared_ptr<const PjRtPlugin>>
  pjrt_plugins_ = {{"LIBRARY", std::make_shared<LibraryPlugin>()}};

template <typename T>
T GetOptionOrEnv(
  const ClientOptions& options, const char* env_var, T default_value);

template <>
bool GetOptionOrEnv<bool>(
  const ClientOptions& options, const char* env_var, bool default_value) {
  auto it = options.find(env_var);
  if (it != options.end()) {
    if (std::holds_alternative<bool>(it->second))
      return std::get<bool>(it->second);
    if (std::holds_alternative<int>(it->second))
      return static_cast<bool>(std::get<int>(it->second));
    if (std::holds_alternative<std::string>(it->second))
      return std::get<std::string>(it->second) == "1";
  }
  return sys_util::GetEnvBool(env_var, default_value);
}

template <>
int GetOptionOrEnv<int>(
  const ClientOptions& options, const char* env_var, int default_value) {
  auto it = options.find(env_var);
  if (it != options.end()) {
    if (std::holds_alternative<int>(it->second))
      return std::get<int>(it->second);
    if (std::holds_alternative<bool>(it->second))
      return static_cast<int>(std::get<bool>(it->second));
    if (std::holds_alternative<std::string>(it->second))
      return std::stoi(std::get<std::string>(it->second));
  }
  return sys_util::GetEnvInt(env_var, default_value);
}

template <>
double GetOptionOrEnv<double>(
  const ClientOptions& options, const char* env_var, double default_value) {
  auto it = options.find(env_var);
  if (it != options.end()) {
    if (std::holds_alternative<int>(it->second))
      return static_cast<double>(std::get<int>(it->second));
    if (std::holds_alternative<bool>(it->second))
      return std::get<bool>(it->second) ? 1.0 : 0.0;
    if (std::holds_alternative<std::string>(it->second))
      return std::stod(std::get<std::string>(it->second));
  }
  return sys_util::GetEnvDouble(env_var, default_value);
}

template <>
std::string GetOptionOrEnv<std::string>(
  const ClientOptions& options, const char* env_var,
  std::string default_value) {
  auto it = options.find(env_var);
  if (it != options.end()) {
    if (std::holds_alternative<std::string>(it->second))
      return std::get<std::string>(it->second);
    if (std::holds_alternative<int>(it->second))
      return std::to_string(std::get<int>(it->second));
    if (std::holds_alternative<bool>(it->second))
      return std::get<bool>(it->second) ? "1" : "0";
  }
  return sys_util::GetEnvString(env_var, default_value);
}

xla::GpuAllocatorConfig GetGpuAllocatorConfig(const ClientOptions& options) {
  auto allocator_config = xla::GpuAllocatorConfig{};
  if (
    GetOptionOrEnv<std::string>(options, env::kEnvPjrtAllocatorCudaAsync, "")
      .empty()
    && GetOptionOrEnv<std::string>(
         options, env::kEnvPjrtAllocatorPreallocate, "")
         .empty()
    && GetOptionOrEnv<std::string>(options, env::kEnvPjrtAllocatorFraction, "")
         .empty()) {
    return allocator_config;
  }
  if (GetOptionOrEnv<bool>(options, env::kEnvPjrtAllocatorCudaAsync, false)) {
    allocator_config.kind = xla::GpuAllocatorConfig::Kind::kCudaAsync;
  }
  allocator_config.preallocate =
    GetOptionOrEnv<bool>(options, env::kEnvPjrtAllocatorPreallocate, false);
  allocator_config.memory_fraction =
    GetOptionOrEnv<double>(options, env::kEnvPjrtAllocatorFraction, 0.75);
  return allocator_config;
}

class CustomDeviceMemoryAllocator : public tsl::Allocator {
 public:
  CustomDeviceMemoryAllocator(
    std::unique_ptr<xla_launcher::AllocatorWrapper> allocator_wrapper)
    : allocator_wrapper_(std::move(allocator_wrapper)) {}
  std::string Name() override { return allocator_wrapper_->Name(); }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    // The order of AllocatorWrapper::Allocate parameters alignment and
    // num_bytes is different from the original tsl::Allocator::AllocateRaw
    // interface.
    return allocator_wrapper_->Allocate(num_bytes, alignment);
  }
  void DeallocateRaw(void* ptr) override {
    allocator_wrapper_->Deallocate(ptr);
  }
  void DeallocateRaw(void* ptr, size_t num_bytes, size_t alignment) override {
    allocator_wrapper_->Deallocate(ptr, num_bytes, alignment);
  }

 private:
  std::unique_ptr<xla_launcher::AllocatorWrapper> allocator_wrapper_;
};

std::unique_ptr<tsl::Allocator> CreateCustomMemoryAllocator(
  int device_ordinal,
  std::function<
    std::unique_ptr<xla_launcher::AllocatorWrapper>(int device_ordinal)>
    allocator_wrapper_factory) {
  return std::make_unique<CustomDeviceMemoryAllocator>(
    allocator_wrapper_factory(device_ordinal));
}

#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

absl::StatusOr<std::unique_ptr<xla::se::GpuCudaMallocAsyncAllocator>>
CreateCudaAsyncAllocator(
  const xla::LocalDeviceState& device, double memory_fraction,
  bool reserve_memory, bool create_new_pool, bool sync_mode,
  bool compute_stats = true) {
  xla::se::StreamExecutor* executor = device.executor();
  int device_ordinal = executor->device_ordinal();

  int64_t free_memory;
  int64_t total_memory;
  if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
    return tsl::errors::Unavailable(
      "Failed to query available memory from device %i", device_ordinal);
  }
  // To allow full GPU memory to be visible to the Cuda Async allocator
  // if using unified memory.
  // When unified memory is enabled, allow GPU memory oversubscription by
  // setting memory_fraction > 1.
  size_t allocator_memory = total_memory * memory_fraction;
  if (reserve_memory) {
    LOG(INFO) << "XLA backend allocating " << allocator_memory
              << " bytes on device " << device_ordinal
              << " for CudaAsyncAllocator.";
  } else {
    LOG(INFO) << "XLA backend will use up to " << allocator_memory
              << " bytes on device " << device_ordinal
              << " for CudaAsyncAllocator.";
  }

  auto allocator = std::make_unique<xla::se::GpuCudaMallocAsyncAllocator>(
    /*platform_device_id*/ tsl::PlatformDeviceId(device_ordinal),
    /*create_new_pool*/ create_new_pool,
    /*new_pool_size*/ allocator_memory,
    /*reserve_memory*/ reserve_memory,
    /*reserve_memory_size*/ reserve_memory ? allocator_memory : 0,
    /*sync_mode*/ sync_mode,
    /*compute_stats*/ compute_stats);

  allocator->SetStreamAndPreallocateMemory(
    device.compute_stream()->platform_specific_handle().stream);

  return allocator;
}

#else  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

absl::StatusOr<std::unique_ptr<tsl::Allocator>> CreateCudaAsyncAllocator(
  const xla::LocalDeviceState& device, double memory_fraction,
  bool reserve_memory, bool create_new_pool, bool sync_mode,
  bool compute_stats = true) {
  return tsl::errors::FailedPrecondition(
    "CUDA async allocator requires CUDA >= 11.2");
}

#endif  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

// Builds a LocalDeviceState for each GPU present.
absl::StatusOr<std::map<int, std::unique_ptr<xla::LocalDeviceState>>>
BuildLocalDeviceStates(xla::LocalClient* xla_client) {
  std::map<int, std::unique_ptr<xla::LocalDeviceState>> addressable_devices;
  for (xla::se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    addressable_devices.emplace(
      executor->device_ordinal(),
      std::make_unique<xla::LocalDeviceState>(
        executor, xla_client, xla::LocalDeviceState::kComputeSynchronized,
        /*max_inflight_computations=*/32,
        /*allow_event_reuse=*/true, /*use_callback_stream=*/true));
  }
  return std::move(addressable_devices);
}

// Constructs a GPU device memory allocator to use, according to the allocator
// configuration the client requested.
absl::StatusOr<std::unique_ptr<xla::se::DeviceMemoryAllocator>>
GetStreamExecutorGpuDeviceAllocator(
  xla::se::Platform* platform, const xla::GpuAllocatorConfig& allocator_config,
  const std::map<int, std::unique_ptr<xla::LocalDeviceState>>&
    addressable_devices,
  std::function<
    std::unique_ptr<xla_launcher::AllocatorWrapper>(int device_ordinal)>
    custom_device_memory_allocator_factory = nullptr) {
  std::vector<xla::se::MultiDeviceAdapter::AllocatorInfo> allocators;
  switch (allocator_config.kind) {
    case xla::GpuAllocatorConfig::Kind::kCudaAsync: {
      for (const auto& ordinal_and_device : addressable_devices) {
        TF_ASSIGN_OR_RETURN(
          auto async_allocator,
          xla_launcher::runtime::CreateCudaAsyncAllocator(
            *(ordinal_and_device.second), allocator_config.memory_fraction,
            allocator_config.preallocate, false, false, true));
        allocators.emplace_back(
          std::move(async_allocator),
          ordinal_and_device.second->compute_stream(),
          /*memory_space=*/0);
      }
      break;
    }

    case xla::GpuAllocatorConfig::Kind::kDefault: {
      if (custom_device_memory_allocator_factory) {
        LOG(INFO) << "Using Custom Device Memory Allocator.";
        for (const auto& ordinal_and_device : addressable_devices) {
          allocators.emplace_back(
            CreateCustomMemoryAllocator(
              ordinal_and_device.first, custom_device_memory_allocator_factory),
            ordinal_and_device.second->compute_stream(),
            /*memory_space=*/0);
        }
        break;
      }
    }
    case xla::GpuAllocatorConfig::Kind::kBFC: {
      LOG(INFO) << "Using BFC allocator.";
      for (const auto& ordinal_and_device : addressable_devices) {
        TF_ASSIGN_OR_RETURN(
          auto bfc_allocator,
          xla::CreateBFCAllocator(
            ordinal_and_device.second->executor(),
            allocator_config.memory_fraction, allocator_config.preallocate,
            allocator_config.gpu_system_memory_size));
        allocators.emplace_back(
          std::move(bfc_allocator), ordinal_and_device.second->compute_stream(),
          /*memory_space=*/0);
      }
      break;
    }

    case xla::GpuAllocatorConfig::Kind::kPlatform:
      LOG(INFO) << "Using platform allocator.";
      if (allocator_config.collective_memory_size != 0) {
        LOG(WARNING)
          << "collective_memory_size is non-zero, but allocator kind is set "
             "to \"platform\". Collective memory will not be allocated.";
      }
      // Returning null will cause the client to use the default backend
      // allocator.
      return nullptr;
  }

  // Add any additional allocators for alternate memory spaces.
  for (const auto& ordinal_and_device : addressable_devices) {
    TF_ASSIGN_OR_RETURN(
      auto collective_bfc_allocator,
      xla::CreateCollectiveBFCAllocator(
        ordinal_and_device.second->executor(),
        /*memory_fraction=*/1.0 - allocator_config.memory_fraction,
        allocator_config.collective_memory_size));
    allocators.emplace_back(
      std::move(collective_bfc_allocator),
      ordinal_and_device.second->compute_stream(),
      /*memory_space=*/1);
  }

  for (const auto& ordinal_and_device : addressable_devices) {
    TF_ASSIGN_OR_RETURN(
      auto host_allocator,
      xla::GetGpuHostAllocator(ordinal_and_device.second->executor()));
    allocators.emplace_back(
      std::move(host_allocator), ordinal_and_device.second->compute_stream(),
      /*memory_space=*/
      static_cast<int>(xla::se::MemoryType::kHost));
  }

#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020
  const auto& debug_options = xla::GetDebugOptionsFromFlags();
  if (debug_options.xla_gpu_temp_buffer_use_separate_color()) {
    // Add memory allocator to allocate memory buffers with persistent temp
    // memory space color.
    for (const auto& ordinal_and_device : addressable_devices) {
      TF_ASSIGN_OR_RETURN(
        auto async_allocator,
        xla_launcher::runtime::CreateCudaAsyncAllocator(
          *(ordinal_and_device.second), 1.0, false, true, true, true));
      allocators.emplace_back(
        std::move(async_allocator), ordinal_and_device.second->compute_stream(),
        /*memory_space=*/xla::gpu::kTempBufferMemorySpaceColor);
    }
  }
#endif
  return std::make_unique<xla::se::MultiDeviceAdapter>(
    platform, std::move(allocators));
}

absl::StatusOr<std::unique_ptr<xla::PjRtClient>> GetStreamExecutorGpuClient(
  const xla::GpuClientOptions& options,
  std::function<
    std::unique_ptr<xla_launcher::AllocatorWrapper>(int device_ordinal)>
    device_memory_allocator_factory = nullptr,
  std::function<
    std::unique_ptr<xla_launcher::AllocatorWrapper>(int device_ordinal)>
    host_memory_allocator_factory = nullptr) {
#if TENSORFLOW_USE_ROCM
  auto pjrt_platform_name = xla::RocmName();
#elif TENSORFLOW_USE_SYCL
  auto pjrt_platform_name = xla::SyclName();
#else   // TENSORFLOW_USE_ROCM
  auto pjrt_platform_name = xla::CudaName();
#endif  // TENSORFLOW_USE_ROCM

  TF_ASSIGN_OR_RETURN(
    xla::LocalClient * xla_client,
    xla::GetGpuXlaClient(options.platform_name, options.allowed_devices));
  std::map<int, std::unique_ptr<xla::LocalDeviceState>> local_device_states;
  TF_ASSIGN_OR_RETURN(
    local_device_states,
    xla_launcher::runtime::BuildLocalDeviceStates(xla_client));
  xla::EnablePeerAccess(xla_client->backend().stream_executors());
  TF_ASSIGN_OR_RETURN(
    auto allocator, xla_launcher::runtime::GetStreamExecutorGpuDeviceAllocator(
                      xla_client->platform(), options.allocator_config,
                      local_device_states, device_memory_allocator_factory));
  std::unique_ptr<tsl::Allocator> host_memory_allocator;
  if (!host_memory_allocator_factory) {
    TF_ASSIGN_OR_RETURN(
      host_memory_allocator,
      xla::GetGpuHostAllocator(
        local_device_states.begin()->second->executor()));
  } else {
    host_memory_allocator =
      CreateCustomMemoryAllocator(0, host_memory_allocator_factory);
  }

  auto gpu_run_options = std::make_unique<xla::gpu::GpuExecutableRunOptions>();
  if (options.enable_mock_nccl) {
    gpu_run_options->set_enable_mock_collectives();
  }

  static const bool xla_gpu_require_exclusive_lock =
    xla::GetDebugOptionsFromFlags().xla_gpu_require_exclusive_lock();
  if (xla_gpu_require_exclusive_lock) {
    gpu_run_options->set_requires_exclusive_lock_on_gpu();
  }

  std::shared_ptr<xla::KeyValueStoreInterface> kv_store = options.kv_store;
  if (options.enable_mock_nccl) {
    kv_store = std::make_shared<xla::InMemoryKeyValueStore>();
  }
  TF_RET_CHECK(options.num_nodes == 1 || kv_store != nullptr);
  TF_ASSIGN_OR_RETURN(
    xla::DeviceTopologyPair device_topology_pair,
    xla::BuildDistributedDevices(
      pjrt_platform_name, std::move(local_device_states), options.node_id,
      options.num_nodes, gpu_run_options.get(), kv_store,
      options.enable_mock_nccl, options.mock_gpu_topology,
      options.slice_index));

  auto gpu_topology = std::shared_ptr<const xla::GpuTopology>(
    xla::GpuTopology::FromProto(device_topology_pair.second));

  return std::unique_ptr<xla::PjRtClient>(
    std::make_unique<xla::StreamExecutorGpuClient>(
      pjrt_platform_name, xla_client, std::move(device_topology_pair.first),
      options.node_id, std::move(allocator), std::move(host_memory_allocator),
      options.should_stage_host_to_device_transfers, std::move(gpu_run_options),
      std::move(kv_store), std::move(options.distributed_runtime_client),
      options.abort_collectives_on_failure, std::move(gpu_topology),
      options.num_nodes));
}

std::shared_ptr<const PjRtPlugin> GetPjRtPlugin(
  const std::string& device_type) {
  auto plugin_path = pjrt_plugins_.find(device_type);
  return plugin_path != pjrt_plugins_.end() ? plugin_path->second : nullptr;
}

}  // namespace

void RegisterPjRtPlugin(
  std::string name, std::shared_ptr<const PjRtPlugin> plugin) {
  VLOG(3) << "Registering PjRt plugin " << name;
  pjrt_plugins_[name] = plugin;
}

std::unique_ptr<xla::PjRtClient> InitializePjRt(
  const ClientOptions& options,
  std::function<
    std::unique_ptr<xla_launcher::AllocatorWrapper>(int device_ordinal)>
    device_memory_allocator_factory,
  std::function<
    std::unique_ptr<xla_launcher::AllocatorWrapper>(int device_ordinal)>
    host_memory_allocator_factory) {
  std::string device_type = absl::AsciiStrToLower(
    GetOptionOrEnv<std::string>(options, env::kEnvPjRtDevice, ""));
  std::unique_ptr<xla::PjRtClient> client;

  if (
    GetOptionOrEnv<bool>(options, env::kEnvPjrtDynamicPlugins, false)
    && device_type != xla::CpuName()) {
    std::shared_ptr<const PjRtPlugin> plugin = GetPjRtPlugin(device_type);
    if (plugin) {
      VLOG(1) << "Initializing client for PjRt plugin " << device_type;

      // Init the absl logging to avoid the log spam.
      absl::InitializeLog();

      std::shared_ptr<xla::KeyValueStoreInterface> kv_store = nullptr;
      if (plugin->requires_xla_coordinator()) {
        XLA_ERROR() << "PjRt plugin " << device_type
                    << " requires XlaCoordinator, which is not supported for "
                       "now.";
        raise(SIGABRT);
      }
      const PJRT_Api* c_api =
        *pjrt::LoadPjrtPlugin(device_type, plugin->library_path());
      XLA_CHECK_OK(pjrt::InitializePjrtPlugin(device_type));
      auto create_options = plugin->client_create_options();
      client =
        xla::GetCApiClient(
          device_type, {create_options.begin(), create_options.end()}, kv_store)
          .value();
      profiler::RegisterProfilerForPlugin(c_api);
    }
  } else if (device_type == xla::CpuName()) {
    VLOG(1) << "Initializing PjRt CPU client...";
    xla::CpuClientOptions cpu_options;
    cpu_options.asynchronous =
      GetOptionOrEnv<bool>(options, env::kEnvPjrtAsyncCpuClient, true);
    cpu_options.cpu_device_count =
      GetOptionOrEnv<int>(options, env::kEnvNumCpu, 1);
    client = std::move(GetXlaPjrtCpuClient(cpu_options).value());
  } else if (device_type == xla::TpuName()) {
    VLOG(1) << "Initializing TFRT TPU client...";
    // Init the absl logging to avoid the log spam.
    absl::InitializeLog();
    // Prefer $TPU_LIBRARY_PATH if set
    auto tpu_library_path = GetOptionOrEnv<std::string>(
      options, env::kEnvTpuLibraryPath,
      GetOptionOrEnv<std::string>(
        options, env::kEnvInferredTpuLibraryPath, "libtpu.so"));
    XLA_CHECK_OK(pjrt::LoadPjrtPlugin("tpu", tpu_library_path).status());
    absl::Status tpu_status = pjrt::InitializePjrtPlugin("tpu");
    XLA_CHECK_OK(tpu_status);
    client = std::move(xla::GetCApiClient(xla::TpuName()).value());
    const PJRT_Api* c_api =
      static_cast<xla::PjRtCApiClient*>(client.get())->pjrt_c_api();
    profiler::RegisterProfilerForPlugin(c_api);
  } else if (
    device_type == xla::CudaName() || device_type == xla::RocmName()
    || device_type == xla::SyclName()) {
    VLOG(1) << "Initializing PjRt " << device_type << " GPU client...";
    bool async =
      GetOptionOrEnv<bool>(options, env::kEnvPjrtAsyncGpuClient, true);
    int local_process_rank =
      GetOptionOrEnv<int>(options, env::kEnvPjRtLocalRank, 0);
    int global_process_rank =
      GetOptionOrEnv<int>(options, env::kEnvPjRtWorldRank, local_process_rank);
    int local_world_size =
      GetOptionOrEnv<int>(options, env::kEnvPjRtLocalSize, 1);
    int global_world_size =
      GetOptionOrEnv<int>(options, env::kEnvPjRtWorldSize, local_world_size);

    VLOG(3) << "Getting StreamExecutorGpuClient for node_id="
            << global_process_rank << ", num_nodes=" << global_world_size
            << ", local_process_rank=" << local_process_rank
            << ", local_world_size=" << local_world_size << ", spmd case="
            << GetOptionOrEnv<bool>(options, env::kEnvXlaUseSpmd, false)
            << ", PJRT_LOCAL_RANK="
            << GetOptionOrEnv<std::string>(options, env::kEnvPjRtLocalRank, "")
            << ", PJRT_WORLD_RANK="
            << GetOptionOrEnv<std::string>(options, env::kEnvPjRtWorldRank, "")
            << ", PJRT_LOCAL_SIZE="
            << GetOptionOrEnv<std::string>(options, env::kEnvPjRtLocalSize, "")
            << ", PJRT_WORLD_SIZE="
            << GetOptionOrEnv<std::string>(options, env::kEnvPjRtWorldSize, "");
    std::optional<std::set<int>> allowed_devices;
    if (local_world_size > 1) {
      allowed_devices = std::set{local_process_rank};
    } else {
      allowed_devices = std::set{0};
    }

    std::shared_ptr<xla::KeyValueStoreInterface> kv_store;
    if (global_world_size > 1) {
      // Use the distributed key-value store from DistributedRuntimeClient.
      // TODO(mofhejia): Implement XLA coordinator distributed key-value store.
    }

    xla::GpuClientOptions gpu_options;
    gpu_options.allocator_config = GetGpuAllocatorConfig(options);
    if (async) {
      gpu_options.allocator_config.kind =
        xla::GpuAllocatorConfig::Kind::kCudaAsync;
    }
    gpu_options.node_id = global_process_rank;
    gpu_options.num_nodes = global_world_size;
    gpu_options.allowed_devices = allowed_devices;
    // When configured on CUDA, "gpu" and "cuda" mean the same thing.
    // When configured on ROCm, "gpu" and "rocm" mean the same thing.
    // When configured on SYCL, "gpu" and "sycl" mean the same thing.
    gpu_options.platform_name = "gpu";
    gpu_options.should_stage_host_to_device_transfers = true;
    gpu_options.kv_store = kv_store;
    client = std::move(xla_launcher::runtime::GetStreamExecutorGpuClient(
                         gpu_options, device_memory_allocator_factory,
                         host_memory_allocator_factory)
                         .value());
  }

  XLA_CHECK(client) << absl::StrFormat(
    "Unknown %s '%s'", env::kEnvPjRtDevice, device_type);

  return client;
}

}  // namespace runtime
}  // namespace xla_launcher
