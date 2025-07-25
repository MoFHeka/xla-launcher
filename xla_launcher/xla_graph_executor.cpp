/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/xla_graph_executor.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tsl/platform/errors.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla_launcher/device.hpp"
#include "xla_launcher/runtime/debug_macros.hpp"
#include "xla_launcher/runtime/runtime.hpp"
#include "xla_launcher/runtime/sys_util.hpp"
#include "xla_launcher/runtime/tensor_source.hpp"
#include "xla_launcher/xla_build_helper.hpp"
#include "xla_launcher/xla_shape_helper.hpp"

namespace xla_launcher {

// static class member initialization
std::once_flag XLAGraphExecutor::init_flag_;
std::unique_ptr<XLAGraphExecutor> XLAGraphExecutor::instance_;
XlaDeviceType XLAGraphExecutor::init_device_type_;
uint64_t XLAGraphExecutor::default_rng_seed_;
std::unordered_map<
  XlaDeviceType, std::unique_ptr<XLAGraphExecutor::ComputationCache>>
  XLAGraphExecutor::computation_cache_;
std::unordered_map<XlaDeviceType, runtime::ComputationClient::DataPtr>
  XLAGraphExecutor::rng_seed_map_;

namespace {
std::unique_ptr<XLAGraphExecutor::ComputationCache> CreateComputationCache(
  XlaDeviceType device_type) {
  static const size_t kMaxCacheSize =
    runtime::sys_util::GetEnvInt("XLA_COMPILATION_CACHE_SIZE", 2048);
  static const bool readonlyPersistentCache =
    runtime::sys_util::GetEnvBool("XLA_PERSISTENT_CACHE_READ_ONLY", false);
  static std::string persistentCacheDir =
    runtime::sys_util::GetEnvString("XLA_PERSISTENT_CACHE_PATH", "");
  if (!persistentCacheDir.empty()) {
    auto serialize_fn =
      [device_type](XLAGraphExecutor::ComputationCache::TypePtr computation)
      -> std::string {
      auto client = runtime::GetComputationClientIfInitialized(device_type);
      if (!client.has_value()) {
        LOG(FATAL) << "Computation client not initialized";
        return "";
      }
      return client.value()->SerializeComputation(computation);
    };
    auto deserialize_fn = [device_type](std::string serialization)
      -> XLAGraphExecutor::ComputationCache::TypePtr {
      auto client = runtime::GetComputationClientIfInitialized(device_type);
      if (!client.has_value()) {
        LOG(FATAL) << "Computation client not initialized";
        return nullptr;
      }
      runtime::ComputationClient::ComputationPtr computation =
        client.value()->DeserializeComputation(serialization);
      if (!computation) {
        LOG(FATAL) << "Failed to deserialize computation";
        return nullptr;
      }
      return computation;
    };
    if (
      runtime::sys_util::GetEnvBool("XLA_HLO_DEBUG", false)
      || runtime::sys_util::GetEnvBool("XLA_IR_DEBUG", false)) {
      LOG(WARNING)
        << "Using persistent compilation cache with XLA_HLO_DEBUG=1 "
           "or XLA_IR_DEBUG=1 is not recommended. Changes to the HLO "
           "metadata will not be reflected in loaded executables.";
    }
    return std::make_unique<XLAGraphExecutor::PersistentCache>(
      kMaxCacheSize, persistentCacheDir, readonlyPersistentCache, serialize_fn,
      deserialize_fn);
  }
  return std::make_unique<XLAGraphExecutor::MemoryCache>(kMaxCacheSize);
}
}  // namespace

void XLAGraphExecutor::Init(
  const runtime::ClientOptions& options,
  DeviceMemoryAllocatorFactory device_memory_allocator_factory,
  HostMemoryAllocatorFactory host_memory_allocator_factory) {
  auto client = runtime::GetComputationClient(
    options, device_memory_allocator_factory, host_memory_allocator_factory);
  if (!client.status().ok()) {
    LOG(FATAL) << "Failed to initialize computation client. Error: "
               << client.status();
    return;
  }
  init_device_type_ = client.value()->GetDeviceType().GetType();
  computation_cache_.insert_or_assign(
    init_device_type_, CreateComputationCache(init_device_type_));
  // Initialize random seed using current time
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;
  default_rng_seed_ = dis(gen);
  for (auto& device : client.value()->GetLocalDevices()) {
    SetRngSeed(device, default_rng_seed_);
  }
}

void XLAGraphExecutor::InitAdditionalComputationClient(
  const runtime::ClientOptions& options,
  DeviceMemoryAllocatorFactory device_memory_allocator_factory,
  HostMemoryAllocatorFactory host_memory_allocator_factory) {
  auto client = runtime::GetComputationClient(
    options, device_memory_allocator_factory, host_memory_allocator_factory);
  if (!client.status().ok()) {
    LOG(FATAL) << "Failed to initialize additional computation client";
    return;
  }
  auto device_type = client.value()->GetDeviceType().GetType();
  auto it = computation_cache_.find(device_type);
  if (it == computation_cache_.end()) {
    computation_cache_.insert_or_assign(
      device_type, CreateComputationCache(device_type));
  } else {
    LOG(WARNING) << "Computation cache already initialized for hardware type: "
                 << DeviceType::XlaDeviceTypeToString(device_type);
  }
  for (auto& device : client.value()->GetLocalDevices()) {
    SetRngSeed(device, default_rng_seed_);
  }
}

bool XLAGraphExecutor::IsComputationCacheInitialized(
  XlaDeviceType device_type) {
  return computation_cache_.find(device_type) != computation_cache_.end();
}

XLAGraphExecutor::ComputationCache* XLAGraphExecutor::GetComputationCache(
  XlaDeviceType device_type) {
  auto it = computation_cache_.find(device_type);
  if (it == computation_cache_.end()) {
    return nullptr;
  }
  return it->second.get();
}

void XLAGraphExecutor::SetRngSeed(
  const std::string& addressable_device, const uint64_t seed) {
  auto literal_seed = xla::LiteralUtil::CreateR0<uint64_t>(seed);
  // Copy inputs to hardware type.
  std::vector<std::shared_ptr<const runtime::TensorSource>> args = {
    std::make_shared<runtime::LiteralSource>(
      std::move(literal_seed), addressable_device),
  };
  std::vector<std::string> parts = absl::StrSplit(addressable_device, ':');
  auto device_type_name = parts[0];
  auto device_type = DeviceType::StringToXlaDeviceType(device_type_name);
  auto client = runtime::GetComputationClientIfInitialized(device_type);
  if (!client.has_value()) {
    LOG(FATAL) << "Computation client not initialized";
    return;
  }
  std::vector<runtime::ComputationClient::DataPtr> data =
    client.value()->TransferToDevice(absl::MakeConstSpan(args));
  rng_seed_map_.insert_or_assign(device_type, std::move(data[0]));
}

runtime::ComputationClient::DataPtr XLAGraphExecutor::GetRngSeed(
  const XlaDeviceType device_type) {
  auto it = rng_seed_map_.find(device_type);
  if (it == rng_seed_map_.end()) {
    return nullptr;
  }
  return it->second;
}

namespace {

inline runtime::ComputationClient::ComputationPtr GetCachedComputation(
  XLAGraphExecutor* executor, hash_util::hash_t hash,
  XlaDeviceType device_type) {
  tsl::profiler::TraceMe activity(
    "GetCachedComputation", tsl::profiler::TraceMeLevel::kInfo);
  auto cache = executor->GetComputationCache(device_type);
  auto computation = cache ? cache->Get(hash) : nullptr;
  VLOG(5) << "Cached computation (hash: " << hash << ")";
  XLA_CHECK(computation) << "Failed to get computation by hash " << hash
                         << ". Maybe the entry get kicked out of the LRU cache";
  return computation;
}

inline runtime::ComputationClient::ComputationPtr CompileStableHloComputation(
  const std::string& stablehlo_bytecode, XlaDeviceType device_type,
  std::function<void(mlir::ModuleOp&, mlir::MLIRContext& context)>
    canonicalize_fn = nullptr) {
  tsl::profiler::TraceMe activity(
    "CompileStableHloComputation", tsl::profiler::TraceMeLevel::kInfo);
  auto client_opt = runtime::GetComputationClientIfInitialized(device_type);
  if (!client_opt.has_value()) {
    LOG(FATAL) << "Computation client not initialized for hardware type "
               << DeviceType::XlaDeviceTypeToString(device_type);
    return nullptr;
  }
  auto client = client_opt.value();
  auto xla_computation =
    client->CompileStableHlo(stablehlo_bytecode, std::move(canonicalize_fn));

  xla::ProgramShape program_shape =
    ConsumeValue(xla_computation.GetProgramShape());
  xla::Shape shape =
    MakeShapeWithDeviceLayout(program_shape.result(), device_type);
  std::string device_type_name = DeviceType::XlaDeviceTypeToString(device_type);

  std::vector<runtime::CompileInstance> instances;
  instances.emplace_back(
    std::move(xla_computation), device_type_name,
    client->GetCompilationDevices(device_type_name, client->GetLocalDevices()),
    &shape);
  std::vector<runtime::ComputationClient::ComputationPtr> computations =
    client->Compile(std::move(instances));
  auto computation = computations[0];
  XLA_CHECK(computation) << "Failed to compile StableHLO computation";
  return computation;
}

template <typename InputT>
struct ParamPrepareTraits;

// For DataPtr vector
template <>
struct ParamPrepareTraits<std::vector<runtime::ComputationClient::DataPtr>> {
  static std::vector<runtime::ComputationClient::DataPtr> Prepare(
    runtime::ComputationClient*, runtime::ComputationClient::ComputationPtr,
    const std::vector<runtime::ComputationClient::DataPtr>& params) {
    return params;
  }
};

// For TensorSource vector
template <>
struct ParamPrepareTraits<
  std::vector<std::shared_ptr<const runtime::TensorSource>>> {
  static std::vector<runtime::ComputationClient::DataPtr> Prepare(
    runtime::ComputationClient* client,
    runtime::ComputationClient::ComputationPtr,
    const std::vector<std::shared_ptr<const runtime::TensorSource>>& params) {
    return client->TransferToDevice(absl::MakeConstSpan(params));
  }
};

// For DataPtr unordered_map
template <>
struct ParamPrepareTraits<
  std::unordered_map<std::string, runtime::ComputationClient::DataPtr>> {
  static std::vector<runtime::ComputationClient::DataPtr> Prepare(
    runtime::ComputationClient*,
    runtime::ComputationClient::ComputationPtr computation,
    const std::unordered_map<std::string, runtime::ComputationClient::DataPtr>&
      params) {
    auto parameter_names = computation->parameter_names();
    assert(parameter_names.size() == params.size());
    std::vector<runtime::ComputationClient::DataPtr> data;
    data.reserve(parameter_names.size());
    for (const auto& name : parameter_names) {
      auto it = params.find(name);
      if (it == params.end()) {
        LOG(FATAL) << "Parameter name not found in graph_inputs: " << name;
        return std::vector<runtime::ComputationClient::DataPtr>();
      }
      data.emplace_back(it->second);
    }
    return data;
  }
};

// For TensorSource unordered_map
template <>
struct ParamPrepareTraits<std::unordered_map<
  std::string, std::shared_ptr<const runtime::TensorSource>>> {
  static std::vector<runtime::ComputationClient::DataPtr> Prepare(
    runtime::ComputationClient* client,
    runtime::ComputationClient::ComputationPtr computation,
    const std::unordered_map<
      std::string, std::shared_ptr<const runtime::TensorSource>>& params) {
    auto parameter_names = computation->parameter_names();
    assert(parameter_names.size() == params.size());
    std::vector<std::shared_ptr<const runtime::TensorSource>> data;
    data.reserve(parameter_names.size());
    for (const auto& name : parameter_names) {
      auto it = params.find(name);
      if (it == params.end()) {
        LOG(FATAL) << "Parameter name not found in graph_inputs: " << name;
        return std::vector<runtime::ComputationClient::DataPtr>();
      }
      data.emplace_back(it->second);
    }
    return client->TransferToDevice(absl::MakeConstSpan(data));
  }
};

template <typename InputT, typename ComputationGetter>
std::unique_ptr<XLAGraphExecutor::Async> ExecuteComputationUnified(
  ComputationGetter get_computation, const InputT& graph_inputs,
  const XlaDeviceType device_type, int local_device_id) {
  tsl::profiler::TraceMe activity(
    "ExecuteComputationUnified", tsl::profiler::TraceMeLevel::kInfo);

  auto computation = get_computation();

  auto client_opt = runtime::GetComputationClientIfInitialized(device_type);
  if (!client_opt.has_value()) {
    LOG(FATAL) << "Computation client not initialized for hardware type "
               << DeviceType::XlaDeviceTypeToString(device_type);
    return nullptr;
  }
  auto client = client_opt.value();
  auto params =
    ParamPrepareTraits<InputT>::Prepare(client, computation, graph_inputs);

  auto async_handle =
    std::make_unique<XLAGraphExecutor::Async>(device_type, params, computation);

  auto* async_handle_ptr = async_handle.get();
  auto async_fn = [async_handle_ptr, client, local_device_id]() {
    tsl::profiler::TraceMe activity(
      "ExecuteComputation_asyncfn", tsl::profiler::TraceMeLevel::kInfo);
    try {
      auto local_device = client->GetLocalDevices()[local_device_id];
      VLOG(3) << "Executing computation on hardware type " << local_device
              << " ...";
      auto results = client->ExecuteComputation(
        *async_handle_ptr->cached_computation, async_handle_ptr->parameters,
        local_device, {});
      return results;
    } catch (const std::exception& e) {
      LOG(ERROR) << "Exception in ExecuteComputation_asyncfn: " << e.what();
      throw;
    }
  };

  async_handle->results = std::async(std::launch::async, async_fn);

  return async_handle;
}

}  // namespace

hash_util::hash_t XLAGraphExecutor::LoadStablehloComputation(
  const std::string& stablehlo_bytecode, const XlaDeviceType device_type,
  std::function<void(mlir::ModuleOp&, mlir::MLIRContext& context)>
    canonicalize_fn) {
  auto computation = CompileStableHloComputation(
    stablehlo_bytecode, device_type, std::move(canonicalize_fn));
  auto cache = GetComputationCache(device_type);
  if (!cache) {
    LOG(FATAL) << "Computation cache not initialized for hardware type "
               << DeviceType::XlaDeviceTypeToString(device_type);
    return hash_util::hash_t();
  }
  auto hash = computation->hash();
  cache->Add(hash, std::move(computation));
  return hash;
}

// Unified ExecuteComputation implementation
template <typename InputT>
std::unique_ptr<XLAGraphExecutor::Async>
XLAGraphExecutor::ExecuteComputationImpl(
  hash_util::hash_t hash, const InputT& graph_inputs,
  const XlaDeviceType device_type, int local_device_id) {
  auto get_computation = [this, &hash, device_type]() {
    return GetCachedComputation(this, hash, device_type);
  };
  return ::xla_launcher::ExecuteComputationUnified(
    get_computation, graph_inputs, device_type, local_device_id);
}

// Unified ExecuteStablehlo implementation
template <typename InputT>
std::unique_ptr<XLAGraphExecutor::Async> XLAGraphExecutor::ExecuteStablehloImpl(
  const std::string& stablehlo_bytecode, const InputT& graph_inputs,
  const XlaDeviceType device_type, int local_device_id) {
  auto get_computation = [stablehlo_bytecode, device_type]() {
    return CompileStableHloComputation(stablehlo_bytecode, device_type);
  };
  return ::xla_launcher::ExecuteComputationUnified(
    get_computation, graph_inputs, device_type, local_device_id);
}

void XLAGraphExecutor::Shutdown() {
  computation_cache_.clear();
  rng_seed_map_.clear();
  runtime::ShutdownAllClients();
}

// Specific interface implementation
std::unique_ptr<XLAGraphExecutor::Async> XLAGraphExecutor::ExecuteComputation(
  hash_util::hash_t hash,
  const std::vector<runtime::ComputationClient::DataPtr>& graph_inputs,
  const XlaDeviceType device_type, int local_device_id) {
  return ExecuteComputationImpl(
    hash, graph_inputs, device_type, local_device_id);
}

std::unique_ptr<XLAGraphExecutor::Async> XLAGraphExecutor::ExecuteComputation(
  hash_util::hash_t hash,
  const std::unordered_map<std::string, runtime::ComputationClient::DataPtr>&
    graph_inputs,
  const XlaDeviceType device_type, int local_device_id) {
  return ExecuteComputationImpl(
    hash, graph_inputs, device_type, local_device_id);
}

std::unique_ptr<XLAGraphExecutor::Async> XLAGraphExecutor::ExecuteComputation(
  hash_util::hash_t hash,
  const std::vector<std::shared_ptr<const runtime::TensorSource>>& graph_inputs,
  const XlaDeviceType device_type, int local_device_id) {
  return ExecuteComputationImpl(
    hash, graph_inputs, device_type, local_device_id);
}

std::unique_ptr<XLAGraphExecutor::Async> XLAGraphExecutor::ExecuteComputation(
  hash_util::hash_t hash,
  const std::unordered_map<
    std::string, std::shared_ptr<const runtime::TensorSource>>& graph_inputs,
  const XlaDeviceType device_type, int local_device_id) {
  return ExecuteComputationImpl(
    hash, graph_inputs, device_type, local_device_id);
}

std::unique_ptr<XLAGraphExecutor::Async> XLAGraphExecutor::ExecuteStablehlo(
  std::string stablehlo_bytecode,
  const std::vector<runtime::ComputationClient::DataPtr>& graph_inputs,
  const XlaDeviceType device_type, int local_device_id) {
  return ExecuteStablehloImpl(
    stablehlo_bytecode, graph_inputs, device_type, local_device_id);
}

std::unique_ptr<XLAGraphExecutor::Async> XLAGraphExecutor::ExecuteStablehlo(
  std::string stablehlo_bytecode,
  const std::unordered_map<std::string, runtime::ComputationClient::DataPtr>&
    graph_inputs,
  const XlaDeviceType device_type, int local_device_id) {
  return ExecuteStablehloImpl(
    stablehlo_bytecode, graph_inputs, device_type, local_device_id);
}

std::unique_ptr<XLAGraphExecutor::Async> XLAGraphExecutor::ExecuteStablehlo(
  std::string stablehlo_bytecode,
  const std::vector<std::shared_ptr<const runtime::TensorSource>>& graph_inputs,
  const XlaDeviceType device_type, int local_device_id) {
  return ExecuteStablehloImpl(
    stablehlo_bytecode, graph_inputs, device_type, local_device_id);
}

std::unique_ptr<XLAGraphExecutor::Async> XLAGraphExecutor::ExecuteStablehlo(
  std::string stablehlo_bytecode,
  const std::unordered_map<
    std::string, std::shared_ptr<const runtime::TensorSource>>& graph_inputs,
  const XlaDeviceType device_type, int local_device_id) {
  return ExecuteStablehloImpl(
    stablehlo_bytecode, graph_inputs, device_type, local_device_id);
}

}  // namespace xla_launcher
