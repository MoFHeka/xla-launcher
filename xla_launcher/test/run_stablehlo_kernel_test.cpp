/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#if GOOGLE_CUDA
#include <cuda_runtime.h>
#endif

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tools/cpp/runfiles/runfiles.h"
#include "xla_launcher/xla_launcher.hpp"

constexpr char NVIDIA_VENDOR_ID[] = "0x10de";

using bazel::tools::cpp::runfiles::Runfiles;
static std::unique_ptr<Runfiles> global_runfiles;

// Check if any NVIDIA GPU exists on Linux by scanning PCI devices.
bool checkNvidiaGpuExistsOnLinux() {
  const std::filesystem::path pci_path("/sys/bus/pci/devices/");
  if (!std::filesystem::exists(pci_path)) {
    std::cerr << "Error: PCI devices path does not exist: " << pci_path
              << std::endl;
    return false;
  }
  try {
    for (const auto& entry : std::filesystem::directory_iterator(pci_path)) {
      std::ifstream vendor(entry.path() / "vendor");
      std::string id;
      if (vendor >> id && id == NVIDIA_VENDOR_ID) {
        return true;
      }
    }
  } catch (const std::filesystem::filesystem_error& e) {
    std::cerr << "Filesystem error: " << e.what() << std::endl;
  }
  return false;
}

namespace xla_launcher {

namespace {
// Aligned memory allocation.
void* AlignedAlloc(size_t size, size_t alignment) {
  void* data = nullptr;
  // posix_memalign is a standard way to get aligned memory on POSIX systems.
  if (posix_memalign(&data, alignment, size) != 0) {
    throw std::runtime_error("Failed to allocate aligned memory.");
  }
  return data;
}

// Helper to read a scalar value from a binary file.
template <typename T>
T ReadScalarFromFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  T value;
  file.read(reinterpret_cast<char*>(&value), sizeof(T));
  return value;
}

// Helper to get total byte size of a tensor.
size_t GetTensorBytes(const DLManagedTensor* tensor) {
  size_t size = 1;
  // Handle scalar case where ndim can be 0.
  if (tensor->dl_tensor.ndim == 0) {
    return (tensor->dl_tensor.dtype.bits / 8);
  }
  for (int i = 0; i < tensor->dl_tensor.ndim; ++i) {
    size *= tensor->dl_tensor.shape[i];
  }
  return size * (tensor->dl_tensor.dtype.bits / 8);
}

// Helper to create a DLManagedTensor from data and shape.
// The created tensor will own the data (host memory).
DLManagedTensor* CreateTensor(
  void* data_ptr, const std::vector<int64_t>& shape, DLDataType dtype,
  DLDevice device) {
  auto* tensor = new DLManagedTensor;
  tensor->manager_ctx = data_ptr;
  tensor->dl_tensor.data = data_ptr;
  tensor->dl_tensor.device = device;
  tensor->dl_tensor.ndim = shape.size();
  tensor->dl_tensor.dtype = dtype;
  tensor->dl_tensor.shape = new int64_t[shape.size()];
  std::copy(shape.begin(), shape.end(), tensor->dl_tensor.shape);
  tensor->dl_tensor.strides = nullptr;  // contiguous
  tensor->dl_tensor.byte_offset = 0;

  tensor->deleter = [](DLManagedTensor* self) {
    if (self) {
      // Use free for memory allocated with AlignedAlloc.
      free(self->dl_tensor.data);
      delete[] self->dl_tensor.shape;
      delete self;
    }
  };

  return tensor;
}

// Helper to read a tensor from a binary file into a host tensor.
DLManagedTensor* ReadTensorFromFile(
  const std::string& path, const std::vector<int64_t>& shape,
  DLDataType dtype) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  constexpr size_t alignment = 64;
  char* data = static_cast<char*>(AlignedAlloc(size, alignment));
  if (!file.read(data, size)) {
    free(data);
    throw std::runtime_error("Failed to read file: " + path);
  }

  size_t expected_elements = 1;
  if (shape.empty()) {  // scalar case
    expected_elements = 1;
  } else {
    for (auto dim : shape) {
      expected_elements *= dim;
    }
  }

  size_t expected_size = expected_elements * (dtype.bits / 8);
  EXPECT_EQ(size, expected_size);

  return CreateTensor(data, shape, dtype, {kDLCPU, 0});
}

#if GOOGLE_CUDA
// Copies a host tensor to the specified device.
DLManagedTensor* CopyTensorToDevice(
  const DLManagedTensor* host_tensor, const DLDevice& device) {
  if (device.device_type == kDLCPU) {
    // For CPU, just create a copy on the host.
    size_t bytes = GetTensorBytes(host_tensor);
    constexpr size_t alignment = 64;
    void* host_data = AlignedAlloc(bytes, alignment);
    memcpy(host_data, host_tensor->dl_tensor.data, bytes);
    std::vector<int64_t> shape(
      host_tensor->dl_tensor.shape,
      host_tensor->dl_tensor.shape + host_tensor->dl_tensor.ndim);
    return CreateTensor(
      host_data, shape, host_tensor->dl_tensor.dtype, {kDLCPU, 0});
  } else if (device.device_type == kDLCUDA) {
    size_t bytes = GetTensorBytes(host_tensor);
    void* device_data;
    cudaMalloc(&device_data, bytes);
    cudaMemcpy(
      device_data, host_tensor->dl_tensor.data, bytes, cudaMemcpyHostToDevice);

    std::vector<int64_t> shape(
      host_tensor->dl_tensor.shape,
      host_tensor->dl_tensor.shape + host_tensor->dl_tensor.ndim);

    auto* tensor = new DLManagedTensor;
    tensor->manager_ctx = nullptr;
    tensor->dl_tensor.data = device_data;
    tensor->dl_tensor.device = device;
    tensor->dl_tensor.ndim = shape.size();
    tensor->dl_tensor.dtype = host_tensor->dl_tensor.dtype;
    tensor->dl_tensor.shape = new int64_t[shape.size()];
    std::copy(shape.begin(), shape.end(), tensor->dl_tensor.shape);
    tensor->dl_tensor.strides = nullptr;  // contiguous
    tensor->dl_tensor.byte_offset = 0;
    tensor->deleter = [](DLManagedTensor* self) {
      if (self) {
        cudaFree(self->dl_tensor.data);
        delete[] self->dl_tensor.shape;
        delete self;
      }
    };
    return tensor;
  }
  throw std::runtime_error("Unsupported device for copy.");
}

// Copies a tensor from any device to the host.
DLManagedTensor* CopyTensorToHost(const DLManagedTensor* device_tensor) {
  size_t bytes = GetTensorBytes(device_tensor);
  constexpr size_t alignment = 64;
  void* host_data = AlignedAlloc(bytes, alignment);

  if (device_tensor->dl_tensor.device.device_type == kDLCPU) {
    memcpy(host_data, device_tensor->dl_tensor.data, bytes);
  } else if (device_tensor->dl_tensor.device.device_type == kDLCUDA) {
    cudaMemcpy(
      host_data, device_tensor->dl_tensor.data, bytes, cudaMemcpyDeviceToHost);
  } else {
    free(host_data);
    throw std::runtime_error("Unsupported device for copy.");
  }

  std::vector<int64_t> shape(
    device_tensor->dl_tensor.shape,
    device_tensor->dl_tensor.shape + device_tensor->dl_tensor.ndim);
  return CreateTensor(
    host_data, shape, device_tensor->dl_tensor.dtype, {kDLCPU, 0});
}
#endif

}  // namespace

using hash_util::hash_t;

class XlaLauncherTest : public ::testing::TestWithParam<std::string> {
 public:
  static void SetUpTestSuite() {
    setenv("PJRT_DEVICE", "cpu", 1);
    launcher_ = std::make_unique<XlaLauncher>(ClientOptions{});
  }

  static void TearDownTestSuite() {
    unsetenv("PJRT_DEVICE");
    launcher_->Shutdown();
  }

 protected:
  void SetUp() override {
    std::string device_type = GetParam();
    bool has_cuda = checkNvidiaGpuExistsOnLinux();
    if (device_type == "cuda" && !has_cuda) {
      GTEST_SKIP() << "No CUDA device available, skipping CUDA test.";
    }
    setenv("PJRT_DEVICE", device_type.c_str(), 1);
    std::cout << "device_type: " << device_type << std::endl;
    // Ensure the launcher is initialized with the correct device type.
    launcher_->InitComputationClient(ClientOptions{});
  }

  static std::unique_ptr<XlaLauncher> launcher_;
  std::map<std::string, std::string> devices_;
  std::map<std::string, hash_t> kernels_hashes_map_;
};

std::unique_ptr<XlaLauncher> XlaLauncherTest::launcher_;

INSTANTIATE_TEST_SUITE_P(
  DeviceTypes, XlaLauncherTest, ::testing::Values("cpu", "cuda"));

// INSTANTIATE_TEST_SUITE_P(
//   DeviceTypes, XlaLauncherTest, ::testing::Values("cuda"));

TEST_P(XlaLauncherTest, LoadStablehloTest) {
  std::string device_type_str = GetParam();
  XlaDeviceType xla_device_type;
  if (device_type_str == "cpu") {
    xla_device_type = XlaDeviceType::CPU;
  } else if (device_type_str == "cuda") {
    xla_device_type = XlaDeviceType::CUDA;
  } else {
    FAIL() << "Unsupported device type: " << device_type_str;
  }
  // Use a fake but syntactically valid StableHLO MLIR string for testing.
  std::string file_content = R"mlir(
module {
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) ->
  tensor<2xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
  )mlir";
  hash_t hash = launcher_->LoadStablehlo(file_content, xla_device_type);
  ASSERT_NE(hash, 0);
}

TEST_P(XlaLauncherTest, RunAdamwKernel) {
  std::string device_type_str = GetParam();
  XlaDeviceType xla_device_type;
  DLDevice device;
  bool is_gpu = false;
  if (device_type_str == "cpu") {
    xla_device_type = XlaDeviceType::CPU;
    device = {kDLCPU, 0};
  } else if (device_type_str == "cuda") {
    xla_device_type = XlaDeviceType::CUDA;
    device = {kDLCUDA, 0};
    is_gpu = true;
  } else {
    FAIL() << "Unsupported device type: " << device_type_str;
  }

  std::string base_path =
    global_runfiles->Rlocation("xla_launcher/xla_launcher/test/adamw_kernel/");
  std::string stablehlo_path = base_path + "adamw.mlir";

  // 1. Load stablehlo file
  std::ifstream file_stream(stablehlo_path, std::ios::in | std::ios::binary);
  ASSERT_TRUE(file_stream.is_open())
    << "Failed to open file: " << stablehlo_path;
  std::string file_content(
    (std::istreambuf_iterator<char>(file_stream)),
    std::istreambuf_iterator<char>());
  file_stream.close();

  // 2. Set up transforms
  ArgumentTransformMap transforms;
  std::string bs_dim_type = "tensor<128x256xf32>";
  transforms["params"] = bs_dim_type;
  transforms["grads"] = bs_dim_type;
  transforms["m"] = bs_dim_type;
  transforms["v"] = bs_dim_type;
  transforms["step"] = std::string("tensor<i32>");
  transforms["lr"] = ReadScalarFromFile<float>(base_path + "lr.bin");
  transforms["b1"] = ReadScalarFromFile<float>(base_path + "b1.bin");
  transforms["b2"] = ReadScalarFromFile<float>(base_path + "b2.bin");
  transforms["eps"] = ReadScalarFromFile<float>(base_path + "eps.bin");
  transforms["weight_decay"] =
    ReadScalarFromFile<float>(base_path + "weight_decay.bin");

  ConstantArgumentTransformMap jax_global_to_constant;
  // Constant values name was defined when adamw.mlir was generated.
  jax_global_to_constant["batch_size"] = static_cast<int32_t>(128);
  jax_global_to_constant["emb_dim"] = static_cast<int32_t>(256);

  hash_t hash = launcher_->LoadStablehlo(
    file_content, transforms, jax_global_to_constant, xla_device_type);
  ASSERT_NE(hash, 0);

  // 3. Load inputs onto host first.
  std::vector<int64_t> tensor_shape = {128, 256};
  std::vector<int64_t> scalar_shape = {};
  DLDataType f32_type = {kDLFloat, 32, 1};

  auto* params_host =
    ReadTensorFromFile(base_path + "params.bin", tensor_shape, f32_type);
  auto* grads_host =
    ReadTensorFromFile(base_path + "grads.bin", tensor_shape, f32_type);
  auto* m_host =
    ReadTensorFromFile(base_path + "m.bin", tensor_shape, f32_type);
  auto* v_host =
    ReadTensorFromFile(base_path + "v.bin", tensor_shape, f32_type);
  auto* step_host =
    ReadTensorFromFile(base_path + "step.bin", scalar_shape, f32_type);

  std::vector<DLManagedTensor*> host_inputs = {
    params_host, grads_host, m_host, v_host, step_host};
  std::vector<DLManagedTensor*> inputs;

  if (is_gpu) {
#if GOOGLE_CUDA
    for (auto* host_tensor : host_inputs) {
      inputs.push_back(CopyTensorToDevice(host_tensor, device));
    }
#endif
  } else {
    inputs = host_inputs;
  }

  // 4. Run computation
  // Remember to move inputs to the function, otherwise memory will be owned by
  // two sides unsafely.
  auto async_results = launcher_->Run(hash, std::move(inputs), xla_device_type);
  std::vector<DLManagedTensor*> device_results = async_results->GetResults();

  // 5. Verify results
  // Copy results to host if computation was on GPU
  std::vector<DLManagedTensor*> results_to_verify;
  if (is_gpu) {
#if GOOGLE_CUDA
    for (auto* device_res : device_results) {
      results_to_verify.push_back(CopyTensorToHost(device_res));
    }
#endif
  } else {
    results_to_verify = device_results;
  }

  ASSERT_EQ(results_to_verify.size(), 4);

  auto* params_new_expected =
    ReadTensorFromFile(base_path + "params_new.bin", tensor_shape, f32_type);
  auto* m_new_expected =
    ReadTensorFromFile(base_path + "m_new.bin", tensor_shape, f32_type);
  auto* v_new_expected =
    ReadTensorFromFile(base_path + "v_new.bin", tensor_shape, f32_type);
  auto* step_new_expected =
    ReadTensorFromFile(base_path + "step_new.bin", scalar_shape, f32_type);

  std::vector<DLManagedTensor*> expected_results = {
    params_new_expected, m_new_expected, v_new_expected, step_new_expected};

  float max_abs_err = 1e-5;
  for (size_t i = 0; i < results_to_verify.size(); ++i) {
    auto* result_tensor = results_to_verify[i];
    auto* expected_tensor = expected_results[i];
    ASSERT_EQ(result_tensor->dl_tensor.ndim, expected_tensor->dl_tensor.ndim);
    size_t num_elements = 1;
    if (result_tensor->dl_tensor.ndim == 0) {
      num_elements = 1;
    } else {
      for (int d = 0; d < result_tensor->dl_tensor.ndim; ++d) {
        ASSERT_EQ(
          result_tensor->dl_tensor.shape[d],
          expected_tensor->dl_tensor.shape[d]);
        num_elements *= result_tensor->dl_tensor.shape[d];
      }
    }

    float* result_data = static_cast<float*>(result_tensor->dl_tensor.data);
    float* expected_data = static_cast<float*>(expected_tensor->dl_tensor.data);

    for (size_t j = 0; j < num_elements; ++j) {
      EXPECT_NEAR(result_data[j], expected_data[j], max_abs_err);
    }
  }

  // Cleanup
  // deleter of inputs has been set to nullptr by FromDLPack, so we don't need
  // to call it here.
#if GOOGLE_CUDA
  if (is_gpu) {
    for (auto t : host_inputs) t->deleter(t);
    for (auto t : results_to_verify) t->deleter(t);
  }
#endif
  for (auto t : device_results) t->deleter(t);
  for (auto t : expected_results) t->deleter(t);
}

}  // namespace xla_launcher

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (!global_runfiles) {
    global_runfiles = std::unique_ptr<Runfiles>(Runfiles::Create(argv[0]));
    assert(global_runfiles != nullptr && "Failed to create runfiles");
  }
  return RUN_ALL_TESTS();
}
