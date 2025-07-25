# XLA Launcher

## 简介

XLA Launcher 是一个高性能、轻量级的C++库，旨在提供一个简洁的接口来加载和执行以 [StableHLO](https://github.com/openxla/stablehlo) 格式表示的计算图。项目受 [pytorch/xla](https://github.com/pytorch/xla) 的启发，致力于将XLA运行时的卓越计算性能，通过C++接口提供给更广泛的应用场景。

我们的目标是让开发者可以轻松地将从 JAX, TensorFlow, PyTorch, ONNX 等主流框架中导出的计算模型，无缝集成到C++应用中，并在CPU、GPU等多种设备上高效运行。

## 核心特性

- **多框架支持**: 无缝加载和执行任何能生成 StableHLO 计算图的框架（如JAX, PyTorch, TensorFlow）所导出的模型。
- **动态尺寸处理**: 支持在运行时将带有动态尺寸（dynamic shapes）的计算图固化（concretize）为静态尺寸，从而实现灵活的模型部署。
- **多设备后端**: 支持在多种硬件上执行计算，目前已支持 CPU 和 NVIDIA GPU。
- **简洁的C++接口**: 提供基于 [DLPack](https://dlpack.ai/spec.html) 的C++接口，实现了与不同框架之间零拷贝的数据交换，简化了集成流程。
- **高性能**: 直接利用XLA的编译和运行时优化能力，提供接近原生的计算性能。

## 快速开始

### 依赖

- [Bazel](https://bazel.build/)
- 支持C++17的编译器
- [OpenXLA](https://github.com/openxla/xla)

### 构建项目

```bash
# 构建所有目标
bazel build //...

# 运行测试
bazel test //...
```

### 打包以便于分发

如需在外部项目中使用 `XlaLauncher`，您可以生成一个包含所需头文件和库文件的分发包。只需运行 `build.sh` 脚本：

```bash
./build.sh
```

该脚本会配置构建环境并创建一个压缩包。成功构建后，您可以在 `bazel-bin/xla_launcher/xla_launcher_tar_gz.tar.gz` 路径下找到该软件包。解压此文件后，即可在您自己的 C++ 应用中链接该库。

### 使用示例

以下是一个简化的示例，展示了如何使用 `XlaLauncher` 加载并执行一个StableHLO模型。

```cpp:path/to/your/main.cpp
#include "xla_launcher/xla_launcher.hpp"
#include <fstream>
#include <iostream>
#include <vector>

int main() {
    // 1. 初始化 XlaLauncher
    xla_launcher::ClientOptions options;
    auto launcher = std::make_unique<xla_launcher::XlaLauncher>(options);
    launcher->InitComputationClient(options);

    // 2. 读取 StableHLO 模型文件
    std::string stablehlo_path = "path/to/your/model.mlir";
    std::ifstream file_stream(stablehlo_path);
    std::string model_content((std::istreambuf_iterator<char>(file_stream)),
                              std::istreambuf_iterator<char>());

    // 3. 定义参数转换，用于固化动态尺寸
    xla_launcher::ArgumentTransformMap transforms;
    // 将模型中名为 "input1" 的参数的形状固化为 tensor<128x256xf32>
    transforms["input1"] = "tensor<128x256xf32>";
    transforms["input2"] = "tensor<128x256xf32>";

    // 定义常量参数转换，用于替换模型中的符号维度
    xla_launcher::ConstantArgumentTransformMap jax_global_to_constant;
    jax_global_to_constant["batch_size"] = static_cast<int32_t>(128);
    jax_global_to_constant["emb_dim"] = static_cast<int32_t>(256);

    // 4. 加载并编译模型
    xla_launcher::XlaDeviceType device_type = xla_launcher::XlaDeviceType::CPU; // Or CUDA
    xla_launcher::hash_t model_hash = launcher->LoadStablehlo(
        model_content, transforms, jax_global_to_constant, device_type);

    // 5. 准备输入数据 (使用 DLManagedTensor)
    // ... 在这里准备你的输入张量 ...
    std::vector<DLManagedTensor*> inputs; 

    // 6. 执行计算
    auto async_results = launcher->Run(model_hash, std::move(inputs), device_type);
    std::vector<DLManagedTensor*> results = async_results->GetResults();

    // 7. 处理输出结果
    // ... 在这里处理你的输出张量 ...
    std::cout << "Execution successful!" << std::endl;

    return 0;
}
```

## 端到端示例：AdamW优化器

为了更好地展示 `XlaLauncher` 的能力，我们提供了一个完整的AdamW优化器示例。该示例涵盖了从生成带有动态尺寸的StableHLO模型，到在C++中加载、固化并执行的全过程。

详细代码请参考：
- **JAX脚本**: [`xla_launcher/test/adamw_kernel/generate_adamw_stablehlo_bin.py`](xla_launcher/test/adamw_kernel/generate_adamw_stablehlo_bin.py)
- **C++测试代码**: [`xla_launcher/test/run_stablehlo_kernel_test.cc`](xla_launcher/test/run_stablehlo_kernel_test.cc)

### 第一步：使用JAX生成带动态尺寸的StableHLO

在Python脚本中，我们使用JAX的 `symbolic_shape` 功能来定义一个具有动态批量大小（`batch_size`）和嵌入维度（`emb_dim`）的AdamW优化器计算图。

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
该脚本会生成以下文件：
- `adamw.mlir`: 包含动态尺寸的StableHLO模型。
- `*.bin`: 输入和期望输出的二进制数据文件，用于测试验证。
- `adamw_signature.yaml`: 描述模型输入输出签名的配置文件。

### 第二步：在C++中加载并执行

在C++测试代码中，我们演示了如何加载`adamw.mlir`，将动态尺寸固化为 `128x256`，并执行计算。

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
这个测试验证了从模型加载、编译到执行的完整流程，并确保计算结果与JAX的原始实现一致。

## 贡献

我们欢迎任何形式的贡献！如果您有任何问题、功能建议或代码改进，请随时提交 [Issues](https://github.com/your-repo/xla_launcher/issues) 或 [Pull Requests](https://github.com/your-repo/xla_launcher/pulls)。

## 许可证

本项目采用 [BSD 3-Clause License](./LICENSE) 许可证。
