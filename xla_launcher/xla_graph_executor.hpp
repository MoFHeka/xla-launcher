/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_XLA_GRAPH_EXECUTOR_HPP_
#define XLA_LAUNCHER_XLA_GRAPH_EXECUTOR_HPP_

#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xla_launcher/hash.hpp"
#include "xla_launcher/runtime/computation_cache.hpp"
#include "xla_launcher/runtime/computation_client.hpp"
#include "xla_launcher/runtime/xla_util.hpp"

namespace xla_launcher {

class XLAGraphExecutor {
 public:
  XLAGraphExecutor(const XLAGraphExecutor&) = delete;
  XLAGraphExecutor& operator=(const XLAGraphExecutor&) = delete;

  XLAGraphExecutor(XLAGraphExecutor&&) = delete;
  XLAGraphExecutor& operator=(XLAGraphExecutor&&) = delete;

  ~XLAGraphExecutor() = default;

  static XLAGraphExecutor& GetInstance(
    runtime::ClientOptions options = {},
    DeviceMemoryAllocatorFactory device_memory_allocator_factory = nullptr,
    HostMemoryAllocatorFactory host_memory_allocator_factory = nullptr) {
    std::call_once(
      init_flag_, &XLAGraphExecutor::Init, options,
      device_memory_allocator_factory, host_memory_allocator_factory);
    return *instance_;
  }

  void InitAdditionalComputationClient(
    const runtime::ClientOptions& options = {},
    DeviceMemoryAllocatorFactory device_memory_allocator_factory = nullptr,
    HostMemoryAllocatorFactory host_memory_allocator_factory = nullptr);

  static void Shutdown();

  using CachedComputation = runtime::Computation;

  using ComputationCache =
    runtime::util::AbstractCache<hash_util::hash_t, CachedComputation>;
  using MemoryCache =
    runtime::util::Cache<hash_util::hash_t, CachedComputation>;
  using PersistentCache =
    runtime::util::PersistentCache<hash_util::hash_t, CachedComputation>;

  ComputationCache* GetComputationCache(XlaDeviceType device_type);

  bool IsComputationCacheInitialized(XlaDeviceType device_type);

  static void SetRngSeed(
    const std::string& addressable_device, const uint64_t seed);

  static runtime::ComputationClient::DataPtr GetRngSeed(
    const XlaDeviceType device_type);

  struct Async {
    XlaDeviceType device_type;
    std::vector<runtime::ComputationClient::DataPtr> parameters;
    std::shared_ptr<XLAGraphExecutor::CachedComputation> cached_computation;
    std::future<std::vector<runtime::ComputationClient::DataPtr>> results;

    Async(
      XlaDeviceType device_type,
      const std::vector<runtime::ComputationClient::DataPtr>& parameters,
      std::shared_ptr<XLAGraphExecutor::CachedComputation> cached_computation)
      : device_type(device_type),
        parameters(parameters),
        cached_computation(std::move(cached_computation)) {}
  };

  hash_util::hash_t LoadStablehloComputation(
    const std::string& stablehlo_bytecode, const XlaDeviceType device_type,
    std::function<void(mlir::ModuleOp&, mlir::MLIRContext& context)>
      canonicalize_fn = nullptr);

  std::unique_ptr<Async> ExecuteComputation(
    hash_util::hash_t hash,
    const std::vector<std::shared_ptr<const runtime::TensorSource>>&
      graph_inputs,
    const XlaDeviceType device_type, int local_device_id);

  std::unique_ptr<Async> ExecuteComputation(
    hash_util::hash_t hash,
    const std::unordered_map<
      std::string, std::shared_ptr<const runtime::TensorSource>>& graph_inputs,
    const XlaDeviceType device_type, int local_device_id);

  std::unique_ptr<Async> ExecuteComputation(
    hash_util::hash_t hash,
    const std::vector<runtime::ComputationClient::DataPtr>& graph_inputs,
    const XlaDeviceType device_type, int local_device_id);

  std::unique_ptr<Async> ExecuteComputation(
    hash_util::hash_t hash,
    const std::unordered_map<std::string, runtime::ComputationClient::DataPtr>&
      graph_inputs,
    const XlaDeviceType device_type, int local_device_id);

  std::unique_ptr<Async> ExecuteStablehlo(
    std::string stablehlo_bytecode,
    const std::vector<std::shared_ptr<const runtime::TensorSource>>&
      graph_inputs,
    const XlaDeviceType device_type, int local_device_id);

  std::unique_ptr<Async> ExecuteStablehlo(
    std::string stablehlo_bytecode,
    const std::unordered_map<
      std::string, std::shared_ptr<const runtime::TensorSource>>& graph_inputs,
    const XlaDeviceType device_type, int local_device_id);

  std::unique_ptr<Async> ExecuteStablehlo(
    std::string stablehlo_bytecode,
    const std::vector<runtime::ComputationClient::DataPtr>& graph_inputs,
    const XlaDeviceType device_type, int local_device_id);

  std::unique_ptr<Async> ExecuteStablehlo(
    std::string stablehlo_bytecode,
    const std::unordered_map<std::string, runtime::ComputationClient::DataPtr>&
      graph_inputs,
    const XlaDeviceType device_type, int local_device_id);

 private:
  XLAGraphExecutor() = default;

  static void Init(
    const runtime::ClientOptions& options = {},
    DeviceMemoryAllocatorFactory device_memory_allocator_factory = nullptr,
    HostMemoryAllocatorFactory host_memory_allocator_factory = nullptr);

  template <typename InputT>
  std::unique_ptr<XLAGraphExecutor::Async> ExecuteComputationImpl(
    hash_util::hash_t hash, const InputT& graph_inputs,
    const XlaDeviceType device_type, int local_device_id);

  template <typename InputT>
  std::unique_ptr<XLAGraphExecutor::Async> ExecuteStablehloImpl(
    const std::string& stablehlo_bytecode, const InputT& graph_inputs,
    const XlaDeviceType device_type, int local_device_id);

  static std::once_flag init_flag_;
  static std::unique_ptr<XLAGraphExecutor> instance_;
  static XlaDeviceType init_device_type_;
  static uint64_t default_rng_seed_;

  static std::unordered_map<XlaDeviceType, std::unique_ptr<ComputationCache>>
    computation_cache_;
  static std::unordered_map<XlaDeviceType, runtime::ComputationClient::DataPtr>
    rng_seed_map_;
};

}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_XLA_GRAPH_EXECUTOR_HPP_
