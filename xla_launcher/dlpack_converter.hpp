/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_DLPACK_CONVERTER_HPP
#define XLA_LAUNCHER_DLPACK_CONVERTER_HPP

#include <dlpack/dlpack.h>

#include "xla_launcher/runtime/computation_client.hpp"

namespace xla_launcher {

DLManagedTensor* ToDLPack(const runtime::ComputationClient::DataPtr& src);
runtime::ComputationClient::DataPtr FromDLPack(DLManagedTensor* src);

}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_DLPACK_CONVERTER_HPP
