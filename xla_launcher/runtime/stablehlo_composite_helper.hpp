/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#pragma once

#ifndef XLA_LAUNCHER_RUNTIME_STABLEHLO_COMPOSITE_HELPER_HPP_
#define XLA_LAUNCHER_RUNTIME_STABLEHLO_COMPOSITE_HELPER_HPP_

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace xla_launcher {
namespace runtime {
namespace util {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateBuildStableHLOCompositePass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateRemoveXlaMarkTensorOpsPass();

}  // namespace util
}  // namespace runtime
}  // namespace xla_launcher

#endif  // XLA_LAUNCHER_RUNTIME_STABLEHLO_COMPOSITE_HELPER_HPP_
