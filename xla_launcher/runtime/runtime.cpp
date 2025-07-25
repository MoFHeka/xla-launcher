/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/runtime/runtime.hpp"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "tsl/platform/stacktrace_handler.h"
#include "xla_launcher/runtime/env.hpp"
#include "xla_launcher/runtime/pjrt_computation_client.hpp"
#include "xla_launcher/runtime/sys_util.hpp"

namespace xla_launcher {
namespace runtime {

// Helper function to create the initialization map
static std::map<xla_launcher::XlaDeviceType, std::atomic_bool>
CreateClientInitializedMap() {
  std::map<xla_launcher::XlaDeviceType, std::atomic_bool> map;
  map.emplace(xla_launcher::XlaDeviceType::CPU, false);
  map.emplace(xla_launcher::XlaDeviceType::CUDA, false);
  map.emplace(xla_launcher::XlaDeviceType::ROCM, false);
  map.emplace(xla_launcher::XlaDeviceType::SYCL, false);
  map.emplace(xla_launcher::XlaDeviceType::TPU, false);
  map.emplace(xla_launcher::XlaDeviceType::PLUGIN, false);
  return map;
}

static std::map<xla_launcher::XlaDeviceType, std::atomic_bool>
  g_client_initialized_map = CreateClientInitializedMap();

static std::map<xla_launcher::XlaDeviceType, std::unique_ptr<ComputationClient>>
  client_instances;

// Global mutex for protecting client initialization
static std::mutex g_client_init_mutex;

namespace {

std::string GetOptionOrEnv(
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

// Creates a new instance of a `ComputationClient` (e.g.
// `PjRtComputationClient`), and initializes the computation client.
// Can only be called when g_computation_client_initialized is false.
static absl::StatusOr<ABSL_ATTRIBUTE_NONNULL(1) const ComputationClient*>
InitializeComputationClient(
  const ClientOptions& options,
  DeviceMemoryAllocatorFactory device_memory_allocator_factory,
  HostMemoryAllocatorFactory host_memory_allocator_factory) {
  // Lock to ensure thread-safe initialization
  std::lock_guard<std::mutex> lock(g_client_init_mutex);

  auto xla_device_type = xla_launcher::DeviceType::StringToXlaDeviceType(
    absl::AsciiStrToLower(GetOptionOrEnv(options, env::kEnvPjRtDevice, "")));

  if (g_client_initialized_map.at(xla_device_type).load()) {
    return client_instances.at(xla_device_type).get();
  }

  if (sys_util::GetEnvBool("XLA_DUMP_FATAL_STACK", false)) {
    tsl::testing::InstallStacktraceHandler();
  }

  // TODO(He Jia): enable IFRT once it's available and stable.
  static bool use_ifrt = sys_util::GetEnvBool("XLA_USE_IFRT", false);
  if (sys_util::GetEnvString(env::kEnvPjRtDevice, "") == "") {
    return absl::FailedPreconditionError("$PJRT_DEVICE is not set.");
  }

  auto device_type = xla_launcher::XlaDeviceType::PLUGIN;
  if (use_ifrt) {
    // auto client_ = std::make_unique<IfrtComputationClient>(options,
    // device_memory_allocator_factory, host_memory_allocator_factory);
    return absl::UnimplementedError(
      "IfrtComputationClient is not implemented.");
  }
  auto client_ = std::make_unique<PjRtComputationClient>(
    options, device_memory_allocator_factory, host_memory_allocator_factory);
  xla_device_type = client_->GetDeviceType().GetType();

  // Double-check pattern to prevent multiple initialization
  if (g_client_initialized_map.at(xla_device_type).load()) {
    return client_instances.at(xla_device_type).get();
  }

  XLA_CHECK(!g_client_initialized_map.at(xla_device_type).load())
    << "InitializeComputationClient() can only be called once.";
  g_client_initialized_map.at(xla_device_type).store(true);
  client_instances.insert_or_assign(xla_device_type, std::move(client_));
  return client_instances.at(xla_device_type).get();
}

}  // namespace

const absl::StatusOr<ABSL_ATTRIBUTE_NONNULL(1) const ComputationClient*>
GetComputationClient(
  ClientOptions options,
  DeviceMemoryAllocatorFactory device_memory_allocator_factory,
  HostMemoryAllocatorFactory host_memory_allocator_factory) {
  return InitializeComputationClient(
    options, device_memory_allocator_factory, host_memory_allocator_factory);
}

std::optional<ABSL_ATTRIBUTE_NONNULL(1) ComputationClient*>
GetComputationClientIfInitialized(
  const xla_launcher::XlaDeviceType device_type) {
  auto g_it = g_client_initialized_map.find(device_type);
  if (g_it != g_client_initialized_map.end()) {
    if (g_it->second.load()) {
      return client_instances.at(device_type).get();
    }
  }
  return std::nullopt;
}

void ShutdownAllClients() {
  // Double-Checked Locking for performance and safety
  static std::atomic_bool shutdown_started(false);
  bool expected = false;
  if (!shutdown_started.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel)) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_client_init_mutex);
  client_instances.clear();
  g_client_initialized_map = CreateClientInitializedMap();
}

}  // namespace runtime
}  // namespace xla_launcher
