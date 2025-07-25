/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_RUNTIME_XLA_ENV_HASH_HPP_
#define XLA_LAUNCHER_RUNTIME_XLA_ENV_HASH_HPP_

#include "xla_launcher/runtime/xla_util.hpp"

namespace xla_launcher {
namespace runtime {
namespace hash {

using hash_t = util::hash_t;

hash_t HashXlaEnvVars();

}  // namespace hash
}  // namespace runtime
}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_XLA_ENV_HASH_HPP_
