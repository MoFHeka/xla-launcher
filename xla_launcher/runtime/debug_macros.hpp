/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_RUNTIME_DEBUG_MACROS_H_
#define XLA_LAUNCHER_RUNTIME_DEBUG_MACROS_H_

#include <utility>

#include "absl/status/status.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/stacktrace.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla_launcher {

#define XLA_ERROR() LOG(ERROR)
#define XLA_CHECK(c) CHECK(c) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_OK(c) TF_CHECK_OK(c) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_EQ(a, b) CHECK_EQ(a, b) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_NE(a, b) CHECK_NE(a, b) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_LE(a, b) CHECK_LE(a, b) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_GE(a, b) CHECK_GE(a, b) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_LT(a, b) CHECK_LT(a, b) << "\n" << tsl::CurrentStackTrace()
#define XLA_CHECK_GT(a, b) CHECK_GT(a, b) << "\n" << tsl::CurrentStackTrace()

template <typename T>
T ConsumeValue(absl::StatusOr<T>&& status) {
  XLA_CHECK_OK(status.status());
  return std::move(status).value();
}

}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_DEBUG_MACROS_H_
