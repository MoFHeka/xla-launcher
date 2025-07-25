/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/xla_launcher.hpp"

#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xla_launcher/dlpack_converter.hpp"
#include "xla_launcher/runtime/runtime.hpp"
#include "xla_launcher/runtime/stablehlo_helper.hpp"
#include "xla_launcher/xla_graph_executor.hpp"

namespace xla_launcher {

class Async::Impl {
 public:
  explicit Impl(std::unique_ptr<XLAGraphExecutor::Async> inner)
    : inner_(std::move(inner)) {}

  std::vector<DLManagedTensor*> GetResults() {
    auto data_ptrs = inner_->results.get();
    std::vector<DLManagedTensor*> outputs;
    outputs.reserve(data_ptrs.size());
    for (auto& d : data_ptrs) {
      outputs.push_back(ToDLPack(d));
    }
    return outputs;
  }

  std::future<std::vector<DLManagedTensor*>> GetFuture() {
    // Wrap the future to convert DataPtr to DLPack
    return std::async(std::launch::async, [this]() {
      auto data_ptrs = inner_->results.get();
      std::vector<DLManagedTensor*> outputs;
      outputs.reserve(data_ptrs.size());
      for (auto& d : data_ptrs) {
        outputs.push_back(ToDLPack(d));
      }
      return outputs;
    });
  }

 private:
  std::unique_ptr<XLAGraphExecutor::Async> inner_;
};

// Async class implementation
Async::Async(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Async::Async(Async&& other) noexcept : impl_(std::move(other.impl_)) {}

Async& Async::operator=(Async&& other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}

Async::~Async() = default;

std::vector<DLManagedTensor*> Async::GetResults() {
  return impl_->GetResults();
}

std::future<std::vector<DLManagedTensor*>> Async::GetFuture() {
  return impl_->GetFuture();
}

class XlaLauncher::Impl {
 public:
  Impl(
    const ClientOptions& options,
    DeviceMemoryAllocatorFactory device_mem_factory,
    HostMemoryAllocatorFactory host_mem_factory)
    : options_(options),
      device_mem_factory_(device_mem_factory),
      host_mem_factory_(host_mem_factory) {
    XLAGraphExecutor::GetInstance(
      options_, device_mem_factory_, host_mem_factory_);
  }

  void InitComputationClient(
    const ClientOptions& options,
    DeviceMemoryAllocatorFactory device_mem_factory,
    HostMemoryAllocatorFactory host_mem_factory) {
    XLAGraphExecutor::GetInstance().InitAdditionalComputationClient(
      options, device_mem_factory, host_mem_factory);
  }

  hash_util::hash_t LoadStablehlo(
    const std::string& stablehlo, const XlaDeviceType device) {
    return XLAGraphExecutor::GetInstance().LoadStablehloComputation(
      stablehlo, device);
  }

  hash_util::hash_t LoadStablehlo(
    const std::string& stablehlo, const ArgumentTransformVec& transforms,
    const ConstantArgumentTransformMap& global_constants,
    const XlaDeviceType device) {
    return XLAGraphExecutor::GetInstance().LoadStablehloComputation(
      stablehlo, device,
      [&](mlir::ModuleOp& mlir_module, mlir::MLIRContext& context) {
        runtime::util::CanonicalizeStableHloStaticShape(
          &mlir_module, &context, transforms, global_constants);
      });
  }

  hash_util::hash_t LoadStablehlo(
    const std::string& stablehlo, const ArgumentTransformLocMap& transforms,
    const ConstantArgumentTransformMap& global_constants,
    const XlaDeviceType device) {
    return XLAGraphExecutor::GetInstance().LoadStablehloComputation(
      stablehlo, device,
      [&](mlir::ModuleOp& mlir_module, mlir::MLIRContext& context) {
        runtime::util::CanonicalizeStableHloStaticShape(
          &mlir_module, &context, transforms, global_constants);
      });
  }

  hash_util::hash_t LoadStablehlo(
    const std::string& stablehlo, const ArgumentTransformMap& transforms,
    const ConstantArgumentTransformMap& global_constants,
    const XlaDeviceType device) {
    return XLAGraphExecutor::GetInstance().LoadStablehloComputation(
      stablehlo, device,
      [&](mlir::ModuleOp& mlir_module, mlir::MLIRContext& context) {
        runtime::util::CanonicalizeStableHloStaticShape(
          &mlir_module, &context, transforms, global_constants);
      });
  }

  // helper function: input conversion
  template <typename InputT, typename OutT>
  static OutT ConvertInputs(InputT&& inputs);

  // helper function: make Async
  static std::unique_ptr<Async> MakeAsync(
    std::unique_ptr<XLAGraphExecutor::Async> async) {
    auto impl = std::make_unique<Async::Impl>(std::move(async));
    return std::unique_ptr<Async>(new Async(std::move(impl)));
  }

  // universal run function
  template <typename InputT, typename GraphInputT, typename ExecutorFn>
  std::unique_ptr<Async> RunImpl(ExecutorFn&& executor, InputT&& inputs) {
    GraphInputT graph_inputs =
      ConvertInputs<InputT, GraphInputT>(std::move(inputs));
    auto async = executor(std::move(graph_inputs));
    return MakeAsync(std::move(async));
  }

 public:
  // stablehlo + vector
  std::unique_ptr<Async> Run(
    const std::string& stablehlo, std::vector<DLManagedTensor*>&& inputs,
    const XlaDeviceType device_type, int local_device_id) {
    return RunImpl<
      std::vector<DLManagedTensor*>,
      std::vector<runtime::ComputationClient::DataPtr>>(
      [&](std::vector<runtime::ComputationClient::DataPtr> graph_inputs) {
        return XLAGraphExecutor::GetInstance().ExecuteStablehlo(
          stablehlo, std::move(graph_inputs), device_type, local_device_id);
      },
      std::move(inputs));
  }

  // stablehlo + unordered_map
  std::unique_ptr<Async> Run(
    const std::string& stablehlo,
    std::unordered_map<std::string, DLManagedTensor*>&& inputs,
    const XlaDeviceType device_type, int local_device_id) {
    return RunImpl<
      std::unordered_map<std::string, DLManagedTensor*>,
      std::unordered_map<std::string, runtime::ComputationClient::DataPtr>>(
      [&](std::unordered_map<std::string, runtime::ComputationClient::DataPtr>
            graph_inputs) {
        return XLAGraphExecutor::GetInstance().ExecuteStablehlo(
          stablehlo, std::move(graph_inputs), device_type, local_device_id);
      },
      std::move(inputs));
  }

  // hash + vector
  std::unique_ptr<Async> Run(
    const hash_util::hash_t& hash, std::vector<DLManagedTensor*>&& inputs,
    const XlaDeviceType device_type, int local_device_id) {
    return RunImpl<
      std::vector<DLManagedTensor*>,
      std::vector<runtime::ComputationClient::DataPtr>>(
      [&](std::vector<runtime::ComputationClient::DataPtr> graph_inputs) {
        return XLAGraphExecutor::GetInstance().ExecuteComputation(
          hash, std::move(graph_inputs), device_type, local_device_id);
      },
      std::move(inputs));
  }

  // hash + unordered_map
  std::unique_ptr<Async> Run(
    const hash_util::hash_t& hash,
    std::unordered_map<std::string, DLManagedTensor*>&& inputs,
    const XlaDeviceType device_type, int local_device_id) {
    return RunImpl<
      std::unordered_map<std::string, DLManagedTensor*>,
      std::unordered_map<std::string, runtime::ComputationClient::DataPtr>>(
      [&](std::unordered_map<std::string, runtime::ComputationClient::DataPtr>
            graph_inputs) {
        return XLAGraphExecutor::GetInstance().ExecuteComputation(
          hash, std::move(graph_inputs), device_type, local_device_id);
      },
      std::move(inputs));
  }

 private:
  ClientOptions options_;
  DeviceMemoryAllocatorFactory device_mem_factory_;
  HostMemoryAllocatorFactory host_mem_factory_;
};

XlaLauncher::XlaLauncher(
  const ClientOptions& options, DeviceMemoryAllocatorFactory device_mem_factory,
  HostMemoryAllocatorFactory host_mem_factory)
  : impl_(
    std::make_unique<Impl>(options, device_mem_factory, host_mem_factory)) {}

XlaLauncher::~XlaLauncher() { Shutdown(); }

void XlaLauncher::InitComputationClient(
  const ClientOptions& options, DeviceMemoryAllocatorFactory device_mem_factory,
  HostMemoryAllocatorFactory host_mem_factory) {
  impl_->InitComputationClient(options, device_mem_factory, host_mem_factory);
}

hash_util::hash_t XlaLauncher::LoadStablehlo(
  const std::string& stablehlo, const XlaDeviceType device) {
  return impl_->LoadStablehlo(stablehlo, device);
}

hash_util::hash_t XlaLauncher::LoadStablehlo(
  const std::string& stablehlo, const ArgumentTransformVec& transforms,
  const ConstantArgumentTransformMap& global_constants,
  const XlaDeviceType device) {
  return impl_->LoadStablehlo(stablehlo, transforms, global_constants, device);
}

hash_util::hash_t XlaLauncher::LoadStablehlo(
  const std::string& stablehlo, const ArgumentTransformLocMap& transforms,
  const ConstantArgumentTransformMap& global_constants,
  const XlaDeviceType device) {
  return impl_->LoadStablehlo(stablehlo, transforms, global_constants, device);
}

hash_util::hash_t XlaLauncher::LoadStablehlo(
  const std::string& stablehlo, const ArgumentTransformMap& transforms,
  const ConstantArgumentTransformMap& global_constants,
  const XlaDeviceType device) {
  return impl_->LoadStablehlo(stablehlo, transforms, global_constants, device);
}

template <>
std::vector<runtime::ComputationClient::DataPtr>
XlaLauncher::Impl::ConvertInputs<
  std::vector<DLManagedTensor*>,
  std::vector<runtime::ComputationClient::DataPtr>>(
  std::vector<DLManagedTensor*>&& inputs) {
  std::vector<runtime::ComputationClient::DataPtr> graph_inputs;
  graph_inputs.reserve(inputs.size());
  for (auto* t : inputs) {
    graph_inputs.emplace_back(FromDLPack(t));
  }
  return graph_inputs;
}

template <>
std::unordered_map<std::string, runtime::ComputationClient::DataPtr>
XlaLauncher::Impl::ConvertInputs<
  std::unordered_map<std::string, DLManagedTensor*>,
  std::unordered_map<std::string, runtime::ComputationClient::DataPtr>>(
  std::unordered_map<std::string, DLManagedTensor*>&& inputs) {
  std::unordered_map<std::string, runtime::ComputationClient::DataPtr>
    graph_inputs;
  graph_inputs.reserve(2 * inputs.size());
  for (auto& t : inputs) {
    graph_inputs.emplace(t.first, FromDLPack(t.second));
  }
  return graph_inputs;
}

std::unique_ptr<Async> XlaLauncher::Run(
  const std::string& stablehlo, std::vector<DLManagedTensor*>&& inputs,
  const XlaDeviceType device, int local_device_id) {
  return impl_->Run(stablehlo, std::move(inputs), device, local_device_id);
}

std::unique_ptr<Async> XlaLauncher::Run(
  const std::string& stablehlo,
  std::unordered_map<std::string, DLManagedTensor*>&& inputs,
  const XlaDeviceType device, int local_device_id) {
  return impl_->Run(stablehlo, std::move(inputs), device, local_device_id);
}

std::unique_ptr<Async> XlaLauncher::Run(
  const hash_util::hash_t& hash, std::vector<DLManagedTensor*>&& inputs,
  const XlaDeviceType device, int local_device_id) {
  return impl_->Run(hash, std::move(inputs), device, local_device_id);
}

std::unique_ptr<Async> XlaLauncher::Run(
  const hash_util::hash_t& hash,
  std::unordered_map<std::string, DLManagedTensor*>&& inputs,
  const XlaDeviceType device, int local_device_id) {
  return impl_->Run(hash, std::move(inputs), device, local_device_id);
}

void XlaLauncher::Shutdown() { XLAGraphExecutor::Shutdown(); }

}  // namespace xla_launcher
