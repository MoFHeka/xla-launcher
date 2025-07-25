/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/runtime/computation_client.hpp"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "deallocation/transforms/passes.h"
#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo_ext/transforms/passes.h"
#include "transforms/gpu_passes.h"
#include "transforms/passes.h"
#include "xla/backends/cpu/codegen/emitters/ir/xla_cpu_dialect.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla_launcher/runtime/mlir_pass_helper.hpp"
#include "xla_launcher/runtime/stablehlo_helper.hpp"

namespace xla_launcher {
namespace runtime {

namespace {
void RegisterPasses(mlir::MLIRContext& context) {
  context.loadDialect<
    mlir::stablehlo::StablehloDialect, mlir::DLTIDialect,
    mlir::affine::AffineDialect, mlir::arith::ArithDialect,
    mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
    mlir::math::MathDialect, xla::cpu::XlaCpuDialect, mlir::mhlo::MhloDialect,
    mlir::scf::SCFDialect, mlir::LLVM::LLVMDialect, mlir::tensor::TensorDialect,
    mlir::vector::VectorDialect, xla::XlaDialect>();

  mlir::registerAllPasses();
  mlir::deallocation::registerDeallocationPasses();
  mlir::hlo::registerLMHLOTransformsPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::registerLMHLOGPUTransformsPasses();
  mlir::stablehlo_ext::registerPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::func::registerInlinerExtension(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::sdy::SdyDialect>();
  context.appendDialectRegistry(registry);
}

xla::XlaComputation CompileStableHloMlirModule(
  mlir::ModuleOp& mlir_module, mlir::MLIRContext& context) {
  xla::HloProto hlo_proto;
  util::ConvertStableHloToHlo(&mlir_module, &context, &hlo_proto);
  xla::HloModuleProto* hlo_module_proto = hlo_proto.mutable_hlo_module();
  xla::XlaComputation computation(*hlo_module_proto);
  return computation;
}
}  // namespace

xla::XlaComputation ComputationClient::CompileStableHlo(
  const std::string& bytecode,
  std::function<void(mlir::ModuleOp&, mlir::MLIRContext& context)>
    canonicalize_fn) {
  mlir::MLIRContext context;
  RegisterPasses(context);

  mlir::OwningOpRef<mlir::ModuleOp> module =
    mlir::stablehlo::deserializePortableArtifact(bytecode, &context);
  mlir::ModuleOp mlir_module = *module;
  if (canonicalize_fn) {
    canonicalize_fn(mlir_module, context);
  }
  return CompileStableHloMlirModule(mlir_module, context);
}

std::shared_ptr<Computation> ComputationClient::Compile(
  xla::XlaComputation computation, std::string compilation_device,
  std::vector<std::string> devices, const xla::Shape* output_shape) {
  std::vector<CompileInstance> instances;
  instances.emplace_back(
    std::move(computation), std::move(compilation_device), std::move(devices),
    output_shape);
  std::vector<std::shared_ptr<Computation>> results =
    Compile(std::move(instances));
  return std::move(results[0]);
}

std::vector<std::string> ComputationClient::GetCompilationDevices(
  const std::string& device, absl::Span<const std::string> devices) {
  std::vector<std::string> compilation_devices;
  if (devices.empty()) {
    // Not support replication device for now
    compilation_devices.push_back(device);
  } else {
    compilation_devices.insert(
      compilation_devices.end(), devices.begin(), devices.end());
  }
  return compilation_devices;
}

int64_t ComputationClient::GetDeviceOrdinal(const std::string& device) {
  auto pos = device.rfind(':');
  XLA_CHECK_NE(pos, std::string::npos) << device;
  return std::stoi(device.substr(pos + 1));
}

}  // namespace runtime
}  // namespace xla_launcher
