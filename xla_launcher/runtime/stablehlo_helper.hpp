/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#ifndef XLA_LAUNCHER_RUNTIME_STABLEHLO_HELPER_HPP_
#define XLA_LAUNCHER_RUNTIME_STABLEHLO_HELPER_HPP_

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "xla/hlo/builder/xla_computation.h"
#include "xla_launcher/runtime/mlir_pass_helper.hpp"

namespace mlir {
class ModuleOp;
class MLIRContext;
class Diagnostic;
class PassManager;
namespace detail {
struct PassExecutionState;
}
}  // namespace mlir

namespace xla_launcher {
namespace runtime {
namespace util {

class ScopedDiagnosticHandler {
 public:
  explicit ScopedDiagnosticHandler(mlir::MLIRContext* context);
  ~ScopedDiagnosticHandler();

  std::string getDiagnostics() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

std::string HloToStablehlo(
  const xla::HloModuleProto* proto, bool emit_bytecode);

void ConvertHloToStableHlo(
  const xla::HloModuleProto* proto, mlir::ModuleOp* mlir_module);

void CanonicalizeStableHloStaticShape(
  mlir::ModuleOp* mlir_module, mlir::MLIRContext* context,
  const ArgumentTransformVec& transforms,
  const ConstantArgumentTransformMap& global_constants_map);

void CanonicalizeStableHloStaticShape(
  mlir::ModuleOp* mlir_module, mlir::MLIRContext* context,
  const ArgumentTransformLocMap& transforms,
  const ConstantArgumentTransformMap& global_constants_map);

void CanonicalizeStableHloStaticShape(
  mlir::ModuleOp* mlir_module, mlir::MLIRContext* context,
  const ArgumentTransformMap& transforms_map,
  const ConstantArgumentTransformMap& global_constants_map);

void ConvertStableHloToHlo(
  mlir::ModuleOp* mlir_module, mlir::MLIRContext* context,
  xla::HloProto* hlo_proto);

std::string GetHloModuleStr(const xla::HloModuleProto* proto);

const std::string GetTorchDtypeToStablehloDtype(const std::string& dtype);

const std::unordered_map<xla::PrimitiveType, std::string>&
GetHloDtypeToStablehloDtypeMap();

std::optional<xla::PrimitiveType> GetIntDtypeToHloDtype(
  const std::string& dtype);

}  // namespace util
}  // namespace runtime
}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_STABLEHLO_HELPER_HPP_
