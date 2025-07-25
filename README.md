# XLA Launcher

## Introduction

XLA Launcher is a high-performance, lightweight C++ library designed to provide a simple interface for loading and executing computation graphs represented in the [StableHLO](https://github.com/openxla/stablehlo) format. Inspired by [pytorch/xla](https://github.com/pytorch/xla), the project aims to bring the exceptional computing performance of the XLA runtime to a wider range of applications through a C++ interface.

Our goal is to enable developers to seamlessly integrate computation models exported from mainstream frameworks like JAX, TensorFlow, PyTorch, and ONNX into C++ applications and run them efficiently on various devices, including CPUs and GPUs.

## Core Features

- **Multi-Framework Support**: Seamlessly load and execute models exported from any framework that can generate a StableHLO computation graph (e.g., JAX, PyTorch, TensorFlow).
- **Dynamic Shape Handling**: Supports concretizing computation graphs with dynamic shapes into static shapes at runtime, allowing for flexible model deployment.
- **Multi-Device Backend**: Supports computation on various hardware, currently including CPUs and NVIDIA GPUs.
- **Simple C++ Interface**: Provides a C++ interface based on [DLPack](https://dlpack.ai/spec.html), enabling zero-copy data exchange with different frameworks and simplifying the integration process.
- **High Performance**: Directly leverages XLA's compilation and runtime optimization capabilities to deliver near-native computation performance.

## Getting Started

### Prerequisites

- [Bazel](https://bazel.build/)
- A C++17 compatible compiler
- [OpenXLA](https://github.com/openxla/xla)

### Building the Project

```bash
# Build all targets
bazel build //...

# Run tests
bazel test //...
```

### Packaging for Distribution

To use `XlaLauncher` in an external project, you can generate a distributable package containing the required headers and library files. Simply run the `build.sh` script:

```bash
./build.sh
```

This script configures the build environment and creates a tarball. After a successful build, the package will be located at `bazel-bin/xla_launcher/xla_launcher_tar_gz.tar.gz`. You can then extract this archive and link against the library in your own C++ application.

### Usage Example

Here is a simplified example of how to use `XlaLauncher` to load and execute a StableHLO model.

```cpp:path/to/your/main.cpp
#include "xla_launcher/xla_launcher.hpp"
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    // 1. Initialize XlaLauncher
    xla_launcher::ClientOptions options;
    auto launcher = std::make_unique<xla_launcher::XlaLauncher>(options);
    launcher->InitComputationClient(options);

    // 2. Read the StableHLO model file
    std::string stablehlo_path = "path/to/your/model.mlir";
    std::ifstream file_stream(stablehlo_path);
    std::string model_content((std::istreambuf_iterator<char>(file_stream)),
                              std::istreambuf_iterator<char>());

    // 3. Define argument transformations to concretize dynamic shapes
    xla_launcher::ArgumentTransformMap transforms;
    // Concretize the shape of the parameter named "input1" to tensor<128x256xf32>
    transforms["input1"] = "tensor<128x256xf32>";
    transforms["input2"] = "tensor<128x256xf32>";

    // Define constant argument transformations to replace symbolic dimensions
    xla_launcher::ConstantArgumentTransformMap jax_global_to_constant;
    jax_global_to_constant["batch_size"] = static_cast<int32_t>(128);
    jax_global_to_constant["emb_dim"] = static_cast<int32_t>(256);

    // 4. Load and compile the model
    xla_launcher::XlaDeviceType device_type = xla_launcher::XlaDeviceType::CPU; // Or CUDA
    xla_launcher::hash_t model_hash = launcher->LoadStablehlo(
        model_content, transforms, jax_global_to_constant, device_type);

    // 5. Prepare input data (using DLManagedTensor)
    // ... prepare your input tensors here ...
    std::vector<DLManagedTensor*> inputs; 

    // 6. Execute the computation
    auto async_results = launcher->Run(model_hash, std::move(inputs), device_type);
    std::vector<DLManagedTensor*> results = async_results->GetResults();

    // 7. Process the output results
    // ... process your output tensors here ...
    std::cout << "Execution successful!" << std::endl;

    return 0;
}
```

## End-to-End Example: AdamW Optimizer

To better demonstrate the capabilities of `XlaLauncher`, we provide a complete example of an AdamW optimizer. This example covers the entire process from generating a StableHLO model with dynamic shapes to loading, concretizing, and executing it in C++.

For detailed code, please refer to:
- **JAX Script**: [`xla_launcher/test/adamw_kernel/generate_adamw_stablehlo_bin.py`](xla_launcher/test/adamw_kernel/generate_adamw_stablehlo_bin.py)
- **C++ Test Code**: [`xla_launcher/test/run_stablehlo_kernel_test.cc`](xla_launcher/test/run_stablehlo_kernel_test.cc)

### Step 1: Generate StableHLO with Dynamic Shapes using JAX

In the Python script, we use JAX's `symbolic_shape` feature to define an AdamW optimizer computation graph with a dynamic batch size (`batch_size`) and embedding dimension (`emb_dim`).

```python:xla_launcher/test/adamw_kernel/generate_adamw_stablehlo_bin.py
// ... existing code ...
    # We define symbolic shapes for the tensor arguments.
    symbolic_dims = export.symbolic_shape(
        "batch_size, emb_dim")  # Dynamic shapes name
    e_params = jax.ShapeDtypeStruct(symbolic_dims, jnp.float32)
    e_grads = jax.ShapeDtypeStruct(symbolic_dims, jnp.float32)
// ... existing code ...
    exp = export.export(jitted_kernel)  # No Platforms
    exp_compile = exp(e_params, e_grads, e_m, e_v, e_step, e_lr, e_b1, e_b2,
                      e_eps, e_weight_decay)
    stablehlo_module_str = exp_compile.mlir_module()
// ... existing code ...
```
This script generates the following files:
- `adamw.mlir`: The StableHLO model with dynamic shapes.
- `*.bin`: Binary data files for inputs and expected outputs, used for verification.
- `adamw_signature.yaml`: A configuration file describing the model's input/output signatures.

### Step 2: Load and Execute in C++

In the C++ test code, we demonstrate how to load `adamw.mlir`, concretize the dynamic shapes to `128x256`, and execute the computation.

```cpp:xla_launcher/test/run_stablehlo_kernel_test.cc
// ... existing code ...
  // 2. Set up transforms
  ArgumentTransformMap transforms;
  std::string bs_dim_type = "tensor<128x256xf32>";
  transforms["params"] = bs_dim_type;
  transforms["grads"] = bs_dim_type;
  transforms["m"] = bs_dim_type;
  transforms["v"] = bs_dim_type;
// ... existing code ...
  ConstantArgumentTransformMap jax_global_to_constant;
  // Constant values name was defined when adamw.mlir was generated.
  jax_global_to_constant["batch_size"] = static_cast<int32_t>(128);
  jax_global_to_constant["emb_dim"] = static_cast<int32_t>(256);

  hash_t hash = client_->LoadStablehlo(
    file_content, transforms, jax_global_to_constant, xla_device_type);
  ASSERT_NE(hash, 0);

  // 3. Load inputs onto host first.
// ... existing code ...
  // 4. Run computation
  // Remember to move inputs to the function, otherwise memory will be owned by
  // two sides unsafely.
  auto async_results = client_->Run(hash, std::move(inputs), xla_device_type);
  std::vector<DLManagedTensor*> device_results = async_results->GetResults();

  // 5. Verify results
// ... existing code ...
```
This test validates the complete workflow from model loading and compilation to execution, ensuring that the computation results are consistent with the original JAX implementation.

## Contributing

We welcome contributions of all forms! If you have any questions, feature suggestions, or code improvements, please feel free to submit [Issues](https://github.com/your-repo/xla_launcher/issues) or [Pull Requests](https://github.com/your-repo/xla_launcher/pulls).

## License

This project is licensed under the [BSD 3-Clause License](./LICENSE).
