/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_DEVICE_HPP
#define XLA_LAUNCHER_DEVICE_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <variant>

namespace xla_launcher {

using ClientOptions =
  std::map<std::string, std::variant<std::string, int, bool>>;

enum class XlaDeviceType { CPU, CUDA, ROCM, SYCL, TPU, PLUGIN };

struct DeviceType {
  explicit DeviceType(XlaDeviceType xla_device_type, std::int64_t ordinal = 0)
    : type_name_(XlaDeviceTypeToString(xla_device_type)),
      type_(xla_device_type),
      ordinal_(ordinal) {}
  explicit DeviceType(const std::string& type_name, std::int64_t ordinal = 0)
    : type_name_(type_name),
      type_(StringToXlaDeviceType(type_name)),
      ordinal_(ordinal) {}
  explicit DeviceType(std::string_view type_name, std::int64_t ordinal = 0)
    : type_name_(std::string(type_name)),
      type_(StringToXlaDeviceType(type_name)),
      ordinal_(ordinal) {}

  std::string ToString() const;
  XlaDeviceType GetType() const;
  std::int8_t type() const { return static_cast<std::int8_t>(type_); }
  std::int64_t ordinal() const { return ordinal_; }

  // Using lowercase here because PJRT platform_name function returns lowercase
  static std::string XlaDeviceTypeToString(XlaDeviceType device_type);
  static XlaDeviceType StringToXlaDeviceType(const std::string& type_name);
  static XlaDeviceType StringToXlaDeviceType(const std::string_view type_name);

 private:
  std::string type_name_;
  XlaDeviceType type_;
  std::int64_t ordinal_;
};

class AllocatorWrapper {
 public:
  explicit AllocatorWrapper(
    int device_ordinal, std::string name = "AllocatorWrapper")
    : device_ordinal_(device_ordinal), name_(name) {}
  virtual ~AllocatorWrapper() = default;
  virtual void* Allocate(size_t num_bytes, size_t alignment = 8) = 0;
  virtual void Deallocate(void* ptr) = 0;
  virtual void Deallocate(void* ptr, size_t num_bytes, size_t alignment) {
    (void)alignment;
    (void)num_bytes;
    Deallocate(ptr);
  }
  virtual std::string Name() { return name_; }

 protected:
  int device_ordinal_;
  std::string name_;
};

using DeviceMemoryAllocatorFactory =
  std::function<std::unique_ptr<AllocatorWrapper>(int)>;
using HostMemoryAllocatorFactory =
  std::function<std::unique_ptr<AllocatorWrapper>(int)>;

DeviceType ParseDeviceString(const std::string& device_spec);

// Return true if the physical device type is TPU.
bool CheckTpuDevice(XlaDeviceType device_type);

}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_DEVICE_HPP
