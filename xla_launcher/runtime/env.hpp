/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_RUNTIME_ENV_HPP_
#define XLA_LAUNCHER_RUNTIME_ENV_HPP_

namespace xla_launcher {
namespace runtime {
namespace env {

constexpr const char* const kEnvNumTpu = "TPU_NUM_DEVICES";
constexpr const char* const kEnvNumGpu = "GPU_NUM_DEVICES";
constexpr const char* const kEnvNumCpu = "CPU_NUM_DEVICES";
constexpr const char* const kEnvPjRtDevice = "PJRT_DEVICE";
constexpr const char* const kEnvPjRtTpuMaxInflightComputations =
  "PJRT_TPU_MAX_INFLIGHT_COMPUTATIONS";
constexpr const char* const kEnvPjrtAsyncCpuClient = "PJRT_CPU_ASYNC_CLIENT";
constexpr const char* const kEnvPjrtAsyncGpuClient = "PJRT_GPU_ASYNC_CLIENT";
constexpr const char* const kEnvTpuLibraryPath = "TPU_LIBRARY_PATH";
constexpr const char* const kEnvInferredTpuLibraryPath =
  "INFERRED_TPU_LIBRARY_PATH";
constexpr const char* const kEnvXpuLibraryPath = "XPU_LIBRARY_PATH";
constexpr const char* const kEnvPjRtLocalRank = "PJRT_LOCAL_RANK";
constexpr const char* const kEnvPjRtLocalSize = "PJRT_LOCAL_SIZE";
constexpr const char* const kEnvPjRtWorldRank = "PJRT_WORLD_RANK";
constexpr const char* const kEnvPjRtWorldSize = "PJRT_WORLD_SIZE";
constexpr const char* const kEnvPjrtAllocatorCudaAsync =
  "PJRT_ALLOCATOR_CUDA_ASYNC";
constexpr const char* const kEnvPjrtAllocatorPreallocate =
  "PJRT_ALLOCATOR_PREALLOCATE";
constexpr const char* const kEnvPjrtAllocatorFraction =
  "PJRT_ALLOCATOR_FRACTION";
constexpr const char* const kEnvPjrtDynamicPlugins = "PJRT_DYNAMIC_PLUGINS";
constexpr const char* const kEnvXlaUseSpmd = "XLA_USE_SPMD";

}  // namespace env
}  // namespace runtime
}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_ENV_HPP_
