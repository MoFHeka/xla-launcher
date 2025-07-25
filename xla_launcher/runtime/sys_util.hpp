/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_RUNTIME_SYS_UTIL_HPP_
#define XLA_LAUNCHER_RUNTIME_SYS_UTIL_HPP_

#include <cstdint>
#include <string>

namespace xla_launcher {
namespace runtime {
namespace sys_util {

// Gets the string environmental variable by `name`, or `defval` if unset.
std::string GetEnvString(const char* name, const std::string& defval);

std::string GetEnvOrdinalPath(
  const char* name, const std::string& defval, const int64_t ordinal);

std::string GetEnvOrdinalPath(
  const char* name, const std::string& defval,
  const char* ordinal_env = "XRT_SHARD_LOCAL_ORDINAL");

// Gets the integer environmental variable by `name`, or `defval` if unset.
int64_t GetEnvInt(const char* name, int64_t defval);

// Gets the double environmental variable by `name`, or `defval` if unset.
double GetEnvDouble(const char* name, double defval);

// Gets the boolean environmental variable by `name`, or `defval` if unset.
bool GetEnvBool(const char* name, bool defval);

// Retrieves the current EPOCH time in nanoseconds.
int64_t NowNs();

}  // namespace sys_util
}  // namespace runtime
}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_SYS_UTIL_HPP_
