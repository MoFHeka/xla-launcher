/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_RUNTIME_TENSOR_RESOURCE_HPP
#define XLA_LAUNCHER_RUNTIME_TENSOR_RESOURCE_HPP

#include <string>
#include <utility>
#include <vector>

#include "dlpack/dlpack.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla_launcher/runtime/debug_macros.hpp"

namespace xla_launcher {
namespace runtime {

class TensorSource {
 public:
  explicit TensorSource(std::string device) : device_(std::move(device)) {}
  virtual ~TensorSource() = default;

  virtual const void* data() const = 0;

  virtual const xla::Shape& shape() const = 0;

  const std::string& device() const { return device_; }

  virtual std::vector<int64_t> byte_strides() const {
    std::vector<int64_t> byte_strides(shape().dimensions_size());
    XLA_CHECK_OK(
      xla::ShapeUtil::ByteStrides(shape(), absl::MakeSpan(byte_strides)));
    return byte_strides;
  }

  virtual std::vector<int64_t> dimensions() const {
    auto dimensions = shape().dimensions();
    return {dimensions.begin(), dimensions.end()};
  }

  virtual xla::PrimitiveType primitive_type() const {
    return shape().element_type();
  }

 private:
  std::string device_;
};

class LiteralSource : public TensorSource {
 public:
  LiteralSource(xla::Literal literal, std::string device)
    : TensorSource(std::move(device)), literal_(std::move(literal)) {}

  const void* data() const override { return literal_.untyped_data(); }

  const xla::Shape& shape() const override { return literal_.shape(); }

 private:
  xla::Literal literal_;
};

class DlpackSource : public TensorSource {
 public:
  DlpackSource(const DLTensor& dlpack, std::string device)
    : TensorSource(std::move(device)),
      dlpack_(dlpack),
      shape_(ConvertDlpackToXlaShape(dlpack)),
      primitive_type_(DlpackDtypeToPrimitiveType(dlpack.dtype)) {}

  const void* data() const override { return dlpack_.data; }

  const xla::Shape& shape() const override { return shape_; }

  xla::PrimitiveType primitive_type() const override { return primitive_type_; }

  // Return byte strides from DLPack if available, otherwise fallback to
  // xla::Shape
  std::vector<int64_t> byte_strides() const override {
    std::vector<int64_t> strides;
    if (dlpack_.strides != nullptr) {
      // DLPack strides are in element units, need to convert to bytes
      int64_t element_size = (dlpack_.dtype.bits * dlpack_.dtype.lanes + 7) / 8;
      for (int i = 0; i < dlpack_.ndim; ++i) {
        strides.push_back(dlpack_.strides[i] * element_size);
      }
    } else {
      // Fallback: use xla::Shape utility
      strides = TensorSource::byte_strides();
    }
    return strides;
  }

  // Return dimensions directly from DLPack
  std::vector<int64_t> dimensions() const override {
    std::vector<int64_t> dims(dlpack_.ndim);
    for (int i = 0; i < dlpack_.ndim; ++i) {
      dims[i] = dlpack_.shape[i];
    }
    return dims;
  }

 private:
  DLTensor dlpack_;
  xla::Shape shape_;
  xla::PrimitiveType primitive_type_;

  // DLPack dtype to xla::PrimitiveType mapping
  static xla::PrimitiveType DlpackDtypeToPrimitiveType(
    const DLDataType& dtype) {
    // Only handle common types, extend as needed
    if (dtype.code == kDLFloat) {
      if (dtype.bits == 32) return xla::F32;
      if (dtype.bits == 64) return xla::F64;
      if (dtype.bits == 16) return xla::F16;
    } else if (dtype.code == kDLInt) {
      if (dtype.bits == 32) return xla::S32;
      if (dtype.bits == 64) return xla::S64;
      if (dtype.bits == 8) return xla::S8;
      if (dtype.bits == 16) return xla::S16;
    } else if (dtype.code == kDLUInt) {
      if (dtype.bits == 8) return xla::U8;
      if (dtype.bits == 16) return xla::U16;
      if (dtype.bits == 32) return xla::U32;
      if (dtype.bits == 64) return xla::U64;
    }
    // Add more as needed
    XLA_ERROR() << "Unsupported DLPack dtype: code=" << int(dtype.code)
                << " bits=" << int(dtype.bits);
    return xla::PrimitiveType::PRIMITIVE_TYPE_INVALID;
  }

  // Convert DLTensor shape to xla::Shape
  static xla::Shape ConvertDlpackToXlaShape(const DLTensor& dlpack) {
    std::vector<int64_t> dims(dlpack.ndim);
    for (int i = 0; i < dlpack.ndim; ++i) {
      dims[i] = dlpack.shape[i];
    }
    xla::PrimitiveType pt = DlpackDtypeToPrimitiveType(dlpack.dtype);
    return xla::ShapeUtil::MakeShape(pt, dims);
  }
};

}  // namespace runtime
}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_TENSOR_RESOURCE_HPP
