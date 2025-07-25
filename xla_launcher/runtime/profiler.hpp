/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#ifndef XLA_LAUNCHER_RUNTIME_PROFILER_HPP_
#define XLA_LAUNCHER_RUNTIME_PROFILER_HPP_

#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "xla/pjrt/c/pjrt_c_api.h"

namespace xla_launcher {
namespace runtime {
namespace profiler {

class ProfilerServer {
  struct Impl;

 public:
  ProfilerServer();
  ~ProfilerServer();
  void Start(int port);

 private:
  std::unique_ptr<Impl> impl_;
};

// Profiler session implementation is based on OpenXLA, we cannot reuse
// the Python binding since it's using nanobind and torch_xla is using pybind11.
// https://github.com/openxla/xla/blob/main/xla/python/profiler.cc
class TslProfilerSessionWrapper {
 public:
  static std::unique_ptr<TslProfilerSessionWrapper> Create();

  explicit TslProfilerSessionWrapper(
    std::unique_ptr<tsl::ProfilerSession> session)
    : session(std::move(session)) {}

  void Export(
    const std::string& xspace_str, const std::string& tensorboard_dir) const;
  const std::string Stop() const;

 private:
  std::unique_ptr<tsl::ProfilerSession> session;
};

absl::Status Trace(
  const char* service_addr, const char* logdir, int duration_ms,
  int num_tracing_attempts,
  const absl::flat_hash_map<std::string, std::variant<bool, int, std::string>>&
    options);

void RegisterProfilerForPlugin(const PJRT_Api* c_api);

}  // namespace profiler
}  // namespace runtime
}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_PROFILER_HPP_
