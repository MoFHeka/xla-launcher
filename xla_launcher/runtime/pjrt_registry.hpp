/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_RUNTIME_PJRT_REGISTRY_HPP_
#define XLA_LAUNCHER_RUNTIME_PJRT_REGISTRY_HPP_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>

#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla_launcher/device.hpp"

namespace xla_launcher {
namespace runtime {

using ClientOptions =
  std::map<std::string, std::variant<std::string, int, bool>>;

class PjRtPlugin {
 public:
  virtual ~PjRtPlugin() = default;

  virtual std::string library_path() const = 0;

  virtual const std::unordered_map<std::string, xla::PjRtValueType>
  client_create_options() const = 0;

  virtual bool requires_xla_coordinator() const = 0;
};

void RegisterPjRtPlugin(
  std::string name, std::shared_ptr<const PjRtPlugin> plugin);

std::unique_ptr<xla::PjRtClient> InitializePjRt(
  const ClientOptions& options = {},
  std::function<
    std::unique_ptr<xla_launcher::AllocatorWrapper>(int device_ordinal)>
    device_memory_allocator_factory = nullptr,
  std::function<
    std::unique_ptr<xla_launcher::AllocatorWrapper>(int device_ordinal)>
    host_memory_allocator_factory = nullptr);

}  // namespace runtime
}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_PJRT_REGISTRY_HPP_
