/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/xla_shape_helper.hpp"

#include <algorithm>
#include <vector>

#include "xla/hlo/builder/xla_builder.h"
#include "xla_launcher/runtime/debug_macros.hpp"
#include "xla_launcher/runtime/sys_util.hpp"

namespace xla_launcher {

absl::StatusOr<const xla::Shape * absl_nonnull> GetShape(xla::XlaOp op) {
  return op.builder()->GetShapePtr(op);
}

xla::PrimitiveType GetShapeDimensionType() {
  // The shape dimension type is always s32 on TPU or CPU.
  return xla::PrimitiveType::S32;
}

double PaddingFactor(int64_t size, int padding) {
  int rem = static_cast<int>(size % padding);
  double ratio = 0.0;
  if (rem > 0) {
    ratio = static_cast<double>(padding - rem) / static_cast<double>(size);
  }
  return 1.0 + ratio;
}

xla::Shape MakeShapeWithSortedLayout(
  absl::Span<const int64_t> dimensions, xla::PrimitiveType type) {
  // Place bigger dimensions on most minor layout locations.
  std::vector<int64_t> layout(dimensions.size());
  std::iota(layout.rbegin(), layout.rend(), 0);
  std::sort(layout.begin(), layout.end(), [&](int64_t a, int64_t b) {
    return dimensions[a] > dimensions[b];
  });
  return xla::ShapeUtil::MakeShapeWithDenseLayout(type, dimensions, layout);
}

xla::Shape* SetDynamicDimensions(
  xla::Shape* shape, absl::Span<const bool> dynamic_dimensions) {
  if (!dynamic_dimensions.empty()) {
    XLA_CHECK_EQ(dynamic_dimensions.size(), shape->dimensions_size());
    for (size_t i = 0; i < dynamic_dimensions.size(); ++i) {
      shape->set_dynamic_dimension(i, dynamic_dimensions[i]);
    }
  }
  return shape;
}

xla::Shape MakeTensorLayout(
  absl::Span<const int64_t> dimensions,
  absl::Span<const bool> dynamic_dimensions, xla::PrimitiveType type) {
  xla::Shape shape =
    xla::ShapeUtil::MakeShapeWithDescendingLayout(type, dimensions);
  SetDynamicDimensions(&shape, dynamic_dimensions);
  return shape;
}

xla::Shape MakeTpuShape(
  absl::Span<const int64_t> dimensions,
  absl::Span<const bool> dynamic_dimensions, xla::PrimitiveType type) {
  static double max_padding_factor =
    runtime::sys_util::GetEnvDouble("XLA_MAX_PADDING_FACTOR", 1.25);
  xla::Shape shape;
  if (
    PaddingFactor(dimensions[dimensions.size() - 1], 128)
      * PaddingFactor(dimensions[dimensions.size() - 2], 8)
    < max_padding_factor) {
    shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(type, dimensions);
  } else {
    shape = MakeShapeWithSortedLayout(dimensions, type);
  }
  SetDynamicDimensions(&shape, dynamic_dimensions);
  return shape;
}

xla::Shape MakeArrayShapeFromDimensions(
  absl::Span<const int64_t> dimensions,
  absl::Span<const bool> dynamic_dimensions, xla::PrimitiveType type,
  XlaDeviceType device_type) {
  if (dimensions.size() > 1 && CheckTpuDevice(device_type)) {
    return MakeTpuShape(dimensions, dynamic_dimensions, type);
  }
  return MakeTensorLayout(dimensions, dynamic_dimensions, type);
}

xla::Shape MakeShapeWithDeviceLayout(
  const xla::Shape& shape, XlaDeviceType device_type) {
  xla::Shape device_shape(shape);
  xla::ShapeUtil::ForEachMutableSubshape(
    &device_shape, [&](xla::Shape* subshape, const xla::ShapeIndex&) {
      if (subshape->IsArray()) {
        *subshape = MakeArrayShapeFromDimensions(
          subshape->dimensions(), subshape->dynamic_dimensions(),
          subshape->element_type(), device_type);
      }
    });
  return device_shape;
}

}  // namespace xla_launcher
