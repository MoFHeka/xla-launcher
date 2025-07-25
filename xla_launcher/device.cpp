/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/device.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "xla/pjrt/pjrt_compiler.h"
#include "xla_launcher/runtime/debug_macros.hpp"

namespace xla_launcher {

std::string DeviceType::XlaDeviceTypeToString(XlaDeviceType device_type) {
  XLA_CHECK(device_type != XlaDeviceType::PLUGIN) << "PLUGIN type name unknown";

  switch (device_type) {
    case XlaDeviceType::CPU:
      return xla::CpuName();
    case XlaDeviceType::CUDA:
      return xla::CudaName();
    case XlaDeviceType::ROCM:
      return xla::RocmName();
    case XlaDeviceType::SYCL:
      return xla::SyclName();
    case XlaDeviceType::TPU:
      return xla::TpuName();
    default:
      XLA_ERROR() << "Invalid device type";
      return "";
  }
}

XlaDeviceType DeviceType::StringToXlaDeviceType(const std::string& type_name) {
  if (type_name == xla::TpuName()) {
    return XlaDeviceType::TPU;
  } else if (type_name == xla::CpuName()) {
    return XlaDeviceType::CPU;
  } else if (type_name == xla::CudaName()) {
    return XlaDeviceType::CUDA;
  } else if (type_name == xla::RocmName()) {
    return XlaDeviceType::ROCM;
  } else if (type_name == xla::SyclName()) {
    return XlaDeviceType::SYCL;
  }

  return XlaDeviceType::PLUGIN;
}

XlaDeviceType DeviceType::StringToXlaDeviceType(
  const std::string_view type_name) {
  if (type_name == xla::TpuName()) {
    return XlaDeviceType::TPU;
  } else if (type_name == xla::CpuName()) {
    return XlaDeviceType::CPU;
  } else if (type_name == xla::CudaName()) {
    return XlaDeviceType::CUDA;
  } else if (type_name == xla::RocmName()) {
    return XlaDeviceType::ROCM;
  } else if (type_name == xla::SyclName()) {
    return XlaDeviceType::SYCL;
  }

  return XlaDeviceType::PLUGIN;
}

std::string DeviceType::ToString() const {
  return type_name_ + ":" + std::to_string(ordinal_);
}

XlaDeviceType DeviceType::GetType() const {
  return static_cast<XlaDeviceType>(type_);
}

DeviceType ParseDeviceString(const std::string& device_spec) {
  XLA_CHECK(!device_spec.empty()) << "empty device spec";
  XLA_CHECK(device_spec[0] != ':')
    << "No device type in device specification: " << device_spec;
  auto pos = device_spec.rfind(':');
  XLA_CHECK_NE(pos, std::string::npos)
    << "Invalid device specification: " << device_spec;
  std::string_view device_type(device_spec.data(), pos);
  std::string_view device_ordinal(
    device_spec.data() + pos + 1, device_spec.size() - pos - 1);

  int ordinal = std::stoi(device_ordinal.data());
  return DeviceType(device_type, ordinal);
}

bool CheckTpuDevice(XlaDeviceType device_type) {
  if (device_type == XlaDeviceType::TPU) {
    return true;
  }
  return false;
}

}  // namespace xla_launcher
