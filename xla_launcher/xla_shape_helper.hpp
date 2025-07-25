/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_XLA_SHAPE_HELPER_HPP_
#define XLA_LAUNCHER_XLA_SHAPE_HELPER_HPP_

#include "absl/base/nullability.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla_launcher/device.hpp"

namespace xla_launcher {

// Returns the shape of the given XLA operation.
absl::StatusOr<const xla::Shape * absl_nonnull> GetShape(xla::XlaOp op);

xla::PrimitiveType GetShapeDimensionType();

xla::Shape MakeShapeWithDeviceLayout(
  const xla::Shape& shape, XlaDeviceType device_type);

}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_XLA_SHAPE_HELPER_HPP_
