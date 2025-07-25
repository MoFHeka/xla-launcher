/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/runtime/sys_util.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>

namespace xla_launcher {
namespace runtime {
namespace sys_util {

std::string GetEnvString(const char* name, const std::string& defval) {
  const char* env = std::getenv(name);
  return env != nullptr ? env : defval;
}

std::string GetEnvOrdinalPath(
  const char* name, const std::string& defval, const int64_t ordinal) {
  std::string path = GetEnvString(name, defval);
  if (!path.empty()) {
    if (ordinal >= 0) {
      path = path + "." + std::to_string(ordinal);
    }
  }
  return path;
}

std::string GetEnvOrdinalPath(
  const char* name, const std::string& defval, const char* ordinal_env) {
  return GetEnvOrdinalPath(name, defval, GetEnvInt(ordinal_env, -1));
}

int64_t GetEnvInt(const char* name, int64_t defval) {
  const char* env = std::getenv(name);
  return env != nullptr ? std::atol(env) : defval;
}

double GetEnvDouble(const char* name, double defval) {
  const char* env = std::getenv(name);
  return env != nullptr ? std::atof(env) : defval;
}

bool GetEnvBool(const char* name, bool defval) {
  const char* env = std::getenv(name);
  if (env == nullptr) {
    return defval;
  }
  if (std::strcmp(env, "true") == 0) {
    return true;
  }
  if (std::strcmp(env, "false") == 0) {
    return false;
  }
  return std::atoi(env) != 0;
}

int64_t NowNs() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           now.time_since_epoch())
    .count();
}

}  // namespace sys_util
}  // namespace runtime
}  // namespace xla_launcher
