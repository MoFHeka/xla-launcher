/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/xla_convert_ops.hpp"

#include <functional>
#include <optional>
#include <tuple>
#include <vector>

#include "tsl/platform/bfloat16.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal_util.h"
#include "xla/permutation_util.h"
#include "xla/types.h"
#include "xla_launcher/runtime/debug_macros.hpp"
#include "xla_launcher/xla_build_helper.hpp"

namespace xla_launcher {

namespace {

xla::XlaOp CreateRawMask(
  xla::XlaOp op, xla::PrimitiveType type, int64_t size, int64_t narrow_size) {
  uint64_t mask_value =
    (static_cast<uint64_t>(1) << narrow_size * CHAR_BIT) - 1;
  xla::XlaOp mask = XlaHelpers::ScalarValue(mask_value, type, op.builder());
  if (xla::primitive_util::IsSignedIntegralType(type)) {
    // Sign extend the truncation mask.
    xla::XlaOp shift = XlaHelpers::ScalarValue<int32_t>(
      (size - narrow_size) * CHAR_BIT, op.builder());
    mask = (mask << shift) >> shift;
  }
  return mask;
}

xla::XlaOp ConvertData(
  xla::XlaOp op, xla::PrimitiveType type, xla::PrimitiveType narrow_type) {
  if (
    !xla::primitive_util::IsIntegralType(type)
    || !xla::primitive_util::IsIntegralType(narrow_type)) {
    return op;
  }
  int64_t size = xla::ShapeUtil::ByteSizeOfPrimitiveType(type);
  int64_t narrow_size = xla::ShapeUtil::ByteSizeOfPrimitiveType(narrow_type);
  XLA_CHECK_GE(size, narrow_size);
  if (size == narrow_size) {
    return op;
  }
  xla::XlaOp mask = CreateRawMask(op, type, size, narrow_size);
  return op & mask;
}

}  // namespace

xla::XlaOp ConvertTo(
  xla::XlaOp op, xla::PrimitiveType from, xla::PrimitiveType to) {
  if (from == to) {
    return op;
  }
  return xla::ConvertElementType(op, to);
}

xla::XlaOp ConvertToRaw(
  xla::XlaOp op, xla::PrimitiveType from, xla::PrimitiveType raw_from,
  xla::PrimitiveType to, xla::PrimitiveType raw_to) {
  if (from != raw_from) {
    op = ConvertData(op, from, raw_from);
  }
  xla::XlaOp result = ConvertTo(op, from, to);
  return to == raw_to ? result : ConvertData(result, to, raw_to);
}

xla::XlaOp ConvertToNumeric(xla::XlaOp op, xla::PrimitiveType from) {
  // TODO(He Jia): Implement when .
  // if (from == xla::PrimitiveType::PRED) {
  //   op = ConvertTo(
  //     op, from,
  //     MaybeDowncastToXlaDeviceType(xla::PrimitiveType::U8, xla_device));
  // }
  return op;
}

xla::XlaOp ConvertToNumeric(xla::XlaOp op) {
  return ConvertToNumeric(op, XlaHelpers::TypeOfXlaOp(op));
}

// TODO(He Jia): Implement DLDataType support.
// xla::XlaOp CastToScalarType(
//   xla::XlaOp input, DLDataType dtype) {
//   if (dtype) {
//     return ConvertTo(
//       input, XlaHelpers::TypeOfXlaOp(input),
//       MakeXlaPrimitiveType(*dtype, &xla_device));
//   }
//   return ConvertToNumeric(input, XlaHelpers::TypeOfXlaOp(input));
// }

xla::XlaOp CastToScalarType(xla::XlaOp input) {
  return ConvertToNumeric(input, XlaHelpers::TypeOfXlaOp(input));
}

xla::XlaOp MaybeConvertTo(xla::XlaOp input, xla::PrimitiveType type) {
  return XlaHelpers::TypeOfXlaOp(input) != type
           ? xla::ConvertElementType(input, type)
           : input;
}

}  // namespace xla_launcher
