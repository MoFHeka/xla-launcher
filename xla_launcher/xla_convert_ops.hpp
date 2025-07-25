/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_XLA_CONVERT_OPS_HPP_
#define XLA_LAUNCHER_XLA_CONVERT_OPS_HPP_

#include <optional>

#include "xla/hlo/builder/xla_builder.h"
#include "xla/types.h"

namespace xla_launcher {

xla::XlaOp ConvertTo(
  xla::XlaOp op, xla::PrimitiveType from, xla::PrimitiveType to);

xla::XlaOp ConvertToRaw(
  xla::XlaOp op, xla::PrimitiveType from, xla::PrimitiveType raw_from,
  xla::PrimitiveType to, xla::PrimitiveType raw_to);

xla::XlaOp ConvertToNumeric(xla::XlaOp op, xla::PrimitiveType from);

xla::XlaOp ConvertToNumeric(xla::XlaOp op);

// Cast the input to the given dtype. If dtype is null, no-op with the exception
// of predicates, which are converted to 8-bit unsigned integers.
// TODO(He Jia): Implement DLDataType support.
// xla::XlaOp CastToScalarType(
//   xla::XlaOp input, std::optional<DLDataType> dtype);
xla::XlaOp CastToScalarType(xla::XlaOp input);

xla::XlaOp MaybeConvertTo(xla::XlaOp input, xla::PrimitiveType type);

}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_XLA_CONVERT_OPS_HPP_
