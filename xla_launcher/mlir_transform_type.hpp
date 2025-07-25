/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 */

#pragma once

#ifndef XLA_LAUNCHER_RUNTIME_MLIR_TRANSFORM_TYPE_HPP
#define XLA_LAUNCHER_RUNTIME_MLIR_TRANSFORM_TYPE_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace xla_launcher {
namespace runtime {
namespace util {

using ReplaceWithConstantFloat = float;
using ReplaceWithConstantDouble = double;
using ReplaceWithConstantUInt32 = uint32_t;
using ReplaceWithConstantInt32 = int32_t;
using ReplaceWithConstantUInt64 = uint64_t;
using ReplaceWithConstantInt64 = int64_t;
using ReplaceWithConstantBool = bool;
using RefineType = std::string;

using ConstantArgumentTransform = std::variant<
  ReplaceWithConstantFloat, ReplaceWithConstantDouble,
  ReplaceWithConstantUInt32, ReplaceWithConstantInt32,
  ReplaceWithConstantUInt64, ReplaceWithConstantInt64, ReplaceWithConstantBool>;

using RefineArgumentTransform = RefineType;

using ArgumentTransform =
  std::variant<ConstantArgumentTransform, RefineArgumentTransform>;

using ArgumentTransformVec = std::vector<ArgumentTransform>;

using ArgumentTransformLocMap = std::map<uint32_t, ArgumentTransform>;

using ArgumentTransformMap = std::map<std::string, ArgumentTransform>;

using ConstantArgumentTransformLocMap =
  std::map<uint32_t, ConstantArgumentTransform>;

using ConstantArgumentTransformMap =
  std::map<std::string, ConstantArgumentTransform>;

using RefineArgumentTransformLocMap =
  std::map<uint32_t, RefineArgumentTransform>;

using RefineArgumentTransformMap =
  std::map<std::string, RefineArgumentTransform>;
}  // namespace util
}  // namespace runtime
}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_MLIR_TRANSFORM_TYPE_HPP
