/**
 * @file xla_launcher.hpp
 * @brief Interface for XLA StableHLO computation launcher.
 *
 * This header provides a minimal, stable, and modern C++ API for loading and
 * executing StableHLO graphs using DLPack tensors. It is designed for use as a
 * third-party shared library and only depends on DLPack and the C++ standard
 * library. All implementation details are hidden (PIMPL).
 *
 * @copyright
 * BSD 3-Clause License, 2025 He Jia <mofhejia@163.com>
 */

#pragma once

#ifndef XLA_LAUNCHER_XLA_LAUNCHER_HPP_
#define XLA_LAUNCHER_XLA_LAUNCHER_HPP_

#include <dlpack/dlpack.h>

#include <future>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "xla_launcher/device.hpp"
#include "xla_launcher/hash.hpp"
#include "xla_launcher/mlir_transform_type.hpp"

namespace xla_launcher {

using runtime::util::ArgumentTransformLocMap;
using runtime::util::ArgumentTransformMap;
using runtime::util::ArgumentTransformVec;
using runtime::util::ConstantArgumentTransformMap;

/**
 * @brief Asynchronous computation result interface.
 *
 * Provides methods to synchronously or asynchronously retrieve computation
 * results. This class uses PIMPL pattern to avoid virtual function overhead
 * while maintaining header isolation.
 */
class Async {
 public:
  friend class XlaLauncher;

  /**
   * @brief Construct Async with implementation.
   * @param impl Implementation pointer (moved).
   */
  Async(Async&& other) noexcept;
  Async& operator=(Async&& other) noexcept;
  ~Async();

  /**
   * @brief Get computation results (blocking).
   * @return Vector of DLPack tensors (caller owns the returned pointers).
   */
  std::vector<DLManagedTensor*> GetResults();

  /**
   * @brief Get a future for asynchronous result retrieval.
   * @return std::future of vector of DLPack tensors.
   */
  std::future<std::vector<DLManagedTensor*>> GetFuture();

 private:
  // Forward declaration of implementation
  class Impl;
  friend class XlaLauncher;

  std::unique_ptr<Impl> impl_;

  // Private constructor for PIMPL pattern, only callable by friends.
  explicit Async(std::unique_ptr<Impl> impl);

  // Disable copy operations
  Async(const Async&) = delete;
  Async& operator=(const Async&) = delete;
};

/**
 * @brief Main interface for XLA StableHLO computation launching.
 *
 * This class is the entry point for loading and executing StableHLO graphs.
 * All implementation details are hidden; only DLPack and std types are exposed.
 */
class XlaLauncher {
 public:
  /**
   * @brief Construct a new XlaLauncher.
   * @param options Client options (see device.hpp).
   * @param device_mem_factory Optional device memory allocator factory.
   * @param host_mem_factory Optional host memory allocator factory.
   */
  XlaLauncher(
    const ClientOptions& options = {},
    DeviceMemoryAllocatorFactory device_mem_factory = nullptr,
    HostMemoryAllocatorFactory host_mem_factory = nullptr);

  ~XlaLauncher();

  /**
   * @brief Initialize the more computation client with different devices.
   * Parameters are the same as the constructor.
   * @param options Client options (see device.hpp).
   * @param device_mem_factory Optional device memory allocator factory.
   * @param host_mem_factory Optional host memory allocator factory.
   */
  void InitComputationClient(
    const ClientOptions& options = {},
    DeviceMemoryAllocatorFactory device_mem_factory = nullptr,
    HostMemoryAllocatorFactory host_mem_factory = nullptr);

  /**
   * @brief Load a StableHLO string and return its hash.
   *
   * Loads a StableHLO bytecode string without applying any argument transforms
   * or global constant replacements. This overload is suitable when no argument
   * refinement or constant replacement is required.
   *
   * @param stablehlo StableHLO bytecode as a string.
   * @param device Target device type (default: XlaDeviceType::CPU).
   * @return hash_t Unique hash for the loaded computation.
   */
  hash_util::hash_t LoadStablehlo(
    const std::string& stablehlo,
    const XlaDeviceType device = XlaDeviceType::CPU);

  /**
   * @brief Load a StableHLO string and return its hash.
   *
   * Loads a StableHLO bytecode string and applies argument transforms and
   * global constant replacements.
   *
   * @param stablehlo StableHLO bytecode as a string.
   * @param transforms Argument transforms for the main function. Each element
   * is a variant:
   *   - If the value is a std::string (e.g., "tensor<f32>", "tensor<1xf32>",
   * "tensor<?xf32>", "tensor<*xf32>"), it is used as a refine-arguments type
   * for stablehlo-refine-arguments.
   *   - If the value is a constant (e.g., float, int, bool), it is used to
   * replace the corresponding argument with a constant.
   * @param global_constants A map from global constant names in the computation
   * graph to their replacement constant values.
   * @param device Target device type (default: XlaDeviceType::CPU).
   * @return hash_t Unique hash for the loaded computation.
   */
  hash_util::hash_t LoadStablehlo(
    const std::string& stablehlo, const ArgumentTransformVec& transforms,
    const ConstantArgumentTransformMap& global_constants = {},
    const XlaDeviceType device = XlaDeviceType::CPU);

  /**
   * @brief Load a StableHLO string and return its hash.
   *
   * Loads a StableHLO bytecode string and applies argument transforms (by
   * argument index) and global constant replacements.
   *
   * @param stablehlo StableHLO bytecode as a string.
   * @param transforms Argument transforms for the main function, indexed by
   * argument position. Each value is a variant:
   *     - std::string for refine-arguments (e.g., "tensor<f32>",
   * "tensor<1xf32>", etc.).
   *     - Constant value (float, int, bool, etc.) for constant replacement.
   * @param global_constants A map from global constant names in the computation
   * graph to their replacement constant values.
   * @param device Target device type (default: XlaDeviceType::CPU).
   * @return hash_t Unique hash for the loaded computation.
   */
  hash_util::hash_t LoadStablehlo(
    const std::string& stablehlo, const ArgumentTransformLocMap& transforms,
    const ConstantArgumentTransformMap& global_constants = {},
    const XlaDeviceType device = XlaDeviceType::CPU);

  /**
   * @brief Load a StableHLO string and return its hash.
   *
   * Loads a StableHLO bytecode string and applies argument transforms (by
   * argument name) and global constant replacements.
   *
   * @param stablehlo StableHLO bytecode as a string.
   * @param transforms Argument transforms for the main function, indexed by
   * argument name. Each value is a variant:
   *     - std::string for refine-arguments (e.g., "tensor<f32>",
   * "tensor<1xf32>", etc.).
   *     - Constant value (float, int, bool, etc.) for constant replacement.
   * @param global_constants A map from global constant names in the computation
   * graph to their replacement constant values.
   * @param device Target device type (default: XlaDeviceType::CPU).
   * @return hash_t Unique hash for the loaded computation.
   */
  hash_util::hash_t LoadStablehlo(
    const std::string& stablehlo, const ArgumentTransformMap& transforms,
    const ConstantArgumentTransformMap& global_constants = {},
    const XlaDeviceType device = XlaDeviceType::CPU);

  /**
   * @brief Execute a StableHLO string on the specified device.
   * @param stablehlo StableHLO bytecode as string.
   * @param inputs Input DLPack tensors. Ownership of all elements in `inputs`
   * will be transferred.
   * @param device_type Device type (e.g., XlaDeviceType::CPU,
   * XlaDeviceType::CUDA).
   * @param local_device_id The index of the device to run the computation on
   * (i.e., which device among the available devices for the compiled
   * computation).
   * @return Async computation result.
   */
  std::unique_ptr<Async> Run(
    const std::string& stablehlo, std::vector<DLManagedTensor*>&& inputs,
    const XlaDeviceType device_type = XlaDeviceType::CPU,
    int local_device_id = 0);

  /**
   * @brief Execute a StableHLO string on the specified device.
   * @param stablehlo StableHLO bytecode as string.
   * @param inputs Input DLPack tensors. Ownership of all elements in `inputs`
   * will be transferred.
   * @param device_type Device type (e.g., XlaDeviceType::CPU,
   * XlaDeviceType::CUDA).
   * @param local_device_id The index of the device to run the computation on
   * (i.e., which device among the available devices for the compiled
   * computation).
   * @return Async computation result.
   */
  std::unique_ptr<Async> Run(
    const std::string& stablehlo,
    std::unordered_map<std::string, DLManagedTensor*>&& inputs,
    const XlaDeviceType device_type = XlaDeviceType::CPU,
    int local_device_id = 0);

  /**
   * @brief Execute a previously loaded computation by hash on the specified
   * device. The device is determined by the compilation device of the
   * computation corresponding to the given hash.
   * @param hash Computation hash.
   * @param inputs Input DLPack tensors. Ownership of all elements in `inputs`
   * will be transferred.
   * @param device_type Device type (e.g., XlaDeviceType::CPU,
   * XlaDeviceType::CUDA).
   * @param local_device_id The index of the device to run the computation on
   * (i.e., which device among the available devices for the compiled
   * computation).
   * @return Async computation result.
   */
  std::unique_ptr<Async> Run(
    const hash_util::hash_t& hash, std::vector<DLManagedTensor*>&& inputs,
    const XlaDeviceType device_type = XlaDeviceType::CPU,
    int local_device_id = 0);

  /**
   * @brief Execute a previously loaded computation by hash on the specified
   * device.
   * @param hash Computation hash.
   * @param inputs Input DLPack tensors. Ownership of all elements in `inputs`
   * will be transferred.
   * @param device_type Device type (e.g., XlaDeviceType::CPU,
   * XlaDeviceType::CUDA).
   * @param local_device_id The index of the device to run the computation on
   * (i.e., which device among the available devices for the compiled
   * computation).
   * @return Async computation result.
   */
  std::unique_ptr<Async> Run(
    const hash_util::hash_t& hash,
    std::unordered_map<std::string, DLManagedTensor*>&& inputs,
    const XlaDeviceType device_type = XlaDeviceType::CPU,
    int local_device_id = 0);

  /**
   * @brief Shuts down the XLA launcher runtime.
   *
   * This function should be called before the program exits to ensure all
   * resources, especially GPU-related resources, are properly released.
   * This is necessary to avoid crashes during static object destruction
   * when the underlying drivers may already be in an invalid state.
   */
  void Shutdown();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_XLA_LAUNCHER_HPP_
