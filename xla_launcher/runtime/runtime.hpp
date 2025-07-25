/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_RUNTIME_RUNTIME_HPP_
#define XLA_LAUNCHER_RUNTIME_RUNTIME_HPP_

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "xla_launcher/device.hpp"
#include "xla_launcher/runtime/computation_client.hpp"

namespace xla_launcher {
namespace runtime {

// Returns the ComputationClient singleton.
const absl::StatusOr<ABSL_ATTRIBUTE_NONNULL(1) const ComputationClient*>
GetComputationClient(
  ClientOptions options = {},
  DeviceMemoryAllocatorFactory device_memory_allocator_factory = nullptr,
  HostMemoryAllocatorFactory host_memory_allocator_factory = nullptr);

// Returns the ComputationClient singleton if it was successfully initialized.
// Returns a nullptr if the ComputationClient wasn't initialized yet.
// Throws an exception if the ComputationClient was initialized but the
// initialization failed.
std::optional<ABSL_ATTRIBUTE_NONNULL(1) ComputationClient*>
GetComputationClientIfInitialized(
  const xla_launcher::XlaDeviceType device_type);

// Runs the XRT local service, this will block the caller unitl the server
// being stopped.
void RunLocalService(uint64_t service_port);

// Shuts down all computation clients.
void ShutdownAllClients();

}  // namespace runtime
}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_RUNTIME_HPP_
