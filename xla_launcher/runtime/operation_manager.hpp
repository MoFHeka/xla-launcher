/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_RUNTIME_OPERATION_MANAGER_HPP
#define XLA_LAUNCHER_RUNTIME_OPERATION_MANAGER_HPP

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "absl/types/span.h"

namespace xla_launcher {
namespace runtime {

// Track inflight operations for each device.
// 'Shared Ptr' for operations
class OperationManager {
 public:
  OperationManager() = default;
  explicit OperationManager(absl::Span<const std::string>);

  OperationManager(const OperationManager&) = delete;
  OperationManager& operator=(const OperationManager&) = delete;

  OperationManager(OperationManager&&) = default;
  OperationManager& operator=(OperationManager&&) = default;

  class Counter {
   public:
    explicit Counter(const std::string& device) : device_(device) {}

    Counter(const Counter&) = delete;
    Counter& operator=(const Counter&) = delete;

    // Register a new operation. Blocks if `BlockNewOperations` has been called.
    void Increment();

    // Mark an inflight task completed.
    void Decrement();

    // Wait until all operations are complete. Does not block new operations
    // (see BlockNewOperations).
    void Wait();

    // Returns a lock that prevents new operations on the device.
    std::unique_lock<std::shared_mutex> BlockNewOperations();

   private:
    std::string device_;

    std::shared_mutex pending_operations_mu_;
    std::atomic<int64_t> count_{0};

    std::mutex cv_mu_;
    std::condition_variable cv_;
  };

  class OperationTracker {
   public:
    // Register an operation in the `counter_`.
    explicit OperationTracker(Counter* counter);

    // Mark an operation complete in `counter_`.
    ~OperationTracker();

    OperationTracker(const OperationTracker&) = delete;
    OperationTracker& operator=(const OperationTracker&) = delete;

   private:
    std::string device_;
    Counter* counter_;
  };

  // Register a new operation for `device`.
  std::unique_ptr<OperationTracker> StartOperation(std::string device);

  // Wait for all device execution to complete on devices.
  void WaitForDevices(absl::Span<const std::string> devices);

 private:
  std::unordered_map<std::string, Counter> op_counters_;
};

}  // namespace runtime
}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_OPERATION_MANAGER_HPP
