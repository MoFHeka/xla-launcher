/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 */

#pragma once

#ifndef XLA_LAUNCHER_RUNTIME_MLIR_PASS_HELPER_HPP
#define XLA_LAUNCHER_RUNTIME_MLIR_PASS_HELPER_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>

#include "mlir/Pass/Pass.h"
#include "xla_launcher/mlir_transform_type.hpp"

namespace xla_launcher {
namespace runtime {
namespace util {

std::unique_ptr<::mlir::Pass> createReplaceFuncArgWithConstantPass(
  ConstantArgumentTransformLocMap arg_to_value,
  std::vector<std::string> target_function_names);

std::unique_ptr<::mlir::Pass> createReplaceGlobalConstantsPass(
  ConstantArgumentTransformMap global_to_value);

std::unique_ptr<::mlir::Pass> createRemoveShapeAssertionsPass();

}  // namespace util
}  // namespace runtime
}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_MLIR_PASS_HELPER_HPP
