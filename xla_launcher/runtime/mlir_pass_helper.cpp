/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 */

#include "xla_launcher/runtime/mlir_pass_helper.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace xla_launcher {
namespace runtime {
namespace util {
namespace {

mlir::Attribute createConstantAttr(
  ConstantArgumentTransform value, mlir::Block *block,
  mlir::OpBuilder *builder) {
  builder->setInsertionPointToStart(block);

  auto create_attr = [&](auto val) -> mlir::Attribute {
    using T = decltype(val);
    mlir::Type elementType;
    if constexpr (std::is_same_v<T, float>) {
      elementType = builder->getF32Type();
    } else if constexpr (std::is_same_v<T, double>) {
      elementType = builder->getF64Type();
    } else if constexpr (std::is_same_v<T, int32_t>) {
      elementType = builder->getI32Type();
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      elementType = builder->getIntegerType(32, false);
    } else if constexpr (std::is_same_v<T, int64_t>) {
      elementType = builder->getI64Type();
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      elementType = builder->getIntegerType(64, false);
    } else if constexpr (std::is_same_v<T, bool>) {
      elementType = builder->getI1Type();
    } else {
      // Should not happen
      return nullptr;
    }

    mlir::ShapedType constantType =
      mlir::RankedTensorType::get({}, elementType);  // Scalar

    if constexpr (std::is_floating_point_v<T>) {
      return mlir::DenseElementsAttr::get(constantType, llvm::APFloat(val));
    } else if constexpr (std::is_integral_v<T>) {
      if constexpr (std::is_same_v<T, bool>) {
        return mlir::DenseElementsAttr::get(
          constantType, llvm::APInt(1, val, false));
      } else {
        return mlir::DenseElementsAttr::get(
          constantType, llvm::APInt(sizeof(T) * 8, val, std::is_signed_v<T>));
      }
    }
    return nullptr;
  };
  return std::visit(create_attr, value);
}

llvm::LogicalResult checkScalarType(
  ConstantArgumentTransform value, mlir::Type type) {
  auto tensorTy = llvm::dyn_cast<mlir::TensorType>(type);
  if (!tensorTy) return mlir::failure();
  auto check = [&](auto val) {
    using T = decltype(val);
    mlir::Type elementType = tensorTy.getElementType();
    if constexpr (std::is_same_v<T, float>) {
      return elementType.isF32();
    } else if constexpr (std::is_same_v<T, double>) {
      return elementType.isF64();
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return elementType.isInteger(32);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      return elementType.isUnsignedInteger(32);
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return elementType.isInteger(64);
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return elementType.isUnsignedInteger(64);
    } else if constexpr (std::is_same_v<T, bool>) {
      return elementType.isInteger(1);
    }
    return false;
  };

  if (std::visit(check, value)) {
    return mlir::success();
  }
  return mlir::failure();
}

struct ReplaceFuncArgWithConstantPass
  : public mlir::PassWrapper<
      ReplaceFuncArgWithConstantPass, mlir::OperationPass<mlir::ModuleOp>> {
 protected:
  ConstantArgumentTransformLocMap arg_to_value_;
  std::vector<std::string> target_function_names_;

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceFuncArgWithConstantPass)

  ReplaceFuncArgWithConstantPass(
    ConstantArgumentTransformLocMap arg_to_value,
    std::vector<std::string> target_function_names)
    : arg_to_value_(std::move(arg_to_value)),
      target_function_names_(std::move(target_function_names)) {}

  ReplaceFuncArgWithConstantPass(const ReplaceFuncArgWithConstantPass &other)
    : PassWrapper(other) {
    arg_to_value_ = other.arg_to_value_;
    target_function_names_ = other.target_function_names_;
  }

  llvm::StringRef getArgument() const final {
    return "replace-func-arg-with-constant";
  }
  llvm::StringRef getDescription() const final {
    return "Replaces a specified tensor<f32> function argument with a "
           "stablehlo.constant.";
  }

  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<mlir::func::FuncOp>()) {
      runOnFunction(funcOp);
    }
  }

  void runOnFunction(mlir::func::FuncOp funcOp) {
    if (
      !target_function_names_.empty()
      && std::find(
           target_function_names_.begin(), target_function_names_.end(),
           funcOp.getSymName())
           == target_function_names_.end()) {
      return;
    }

    if (funcOp.isExternal() || funcOp.empty()) {
      return;
    }

    mlir::Block &entryBlock = funcOp.getBody().getBlocks().front();
    mlir::MLIRContext *context = funcOp.getContext();

    // Replace all occurences of the arguments with a constant.
    for (const auto &arg : arg_to_value_) {
      unsigned int index = arg.first;
      if (index >= entryBlock.getNumArguments()) {
        llvm::StringRef func_name = funcOp.getSymName();
        funcOp.emitError("Argument index ")
          << index << " is out of bounds " << entryBlock.getNumArguments()
          << " in " << func_name.str() << ".";
        return signalPassFailure();
      }

      mlir::BlockArgument blockArg = entryBlock.getArgument(index);

      if (checkScalarType(arg.second, blockArg.getType()).failed()) {
        funcOp.emitError("argument is not of correct scalar type.");
        return signalPassFailure();
      }

      // Create constant
      mlir::OpBuilder builder(context);
      mlir::Attribute valueAttr =
        createConstantAttr(arg.second, &entryBlock, &builder);
      if (!valueAttr) {
        funcOp.emitError(
          "Failed to create DenseElementsAttr for the constant value.");
        return signalPassFailure();
      }
      mlir::Location loc = funcOp.getLoc();
      auto constantOp =
        builder.create<mlir::stablehlo::ConstantOp>(loc, valueAttr);

      // Replace uses and erase
      blockArg.replaceAllUsesWith(constantOp.getResult());
    }

    llvm::BitVector should_erase(entryBlock.getNumArguments(), false);
    for (const auto &arg : arg_to_value_) {
      should_erase[arg.first] = true;
    }
    if (should_erase.none()) return;
    entryBlock.eraseArguments(should_erase);

    // Update function type
    mlir::FunctionType oldFuncType = funcOp.getFunctionType();
    llvm::SmallVector<mlir::Type, 4> newInputTypes;
    llvm::SmallVector<mlir::DictionaryAttr, 4> newArgAttrs;
    for (unsigned i = 0; i < oldFuncType.getNumInputs(); ++i) {
      if (!should_erase[i]) {
        newInputTypes.push_back(oldFuncType.getInput(i));
        newArgAttrs.push_back(funcOp.getArgAttrDict(i));
      }
    }
    mlir::FunctionType newFuncType =
      mlir::FunctionType::get(context, newInputTypes, oldFuncType.getResults());
    funcOp.setType(newFuncType);
    funcOp.setAllArgAttrs(newArgAttrs);
  }
};

struct ReplaceGlobalConstantsPass
  : public mlir::PassWrapper<
      ReplaceGlobalConstantsPass, mlir::OperationPass<mlir::ModuleOp>> {
 protected:
  ConstantArgumentTransformMap global_to_value_;

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceGlobalConstantsPass)

  explicit ReplaceGlobalConstantsPass(
    ConstantArgumentTransformMap global_to_value)
    : global_to_value_(std::move(global_to_value)) {}

  ReplaceGlobalConstantsPass(const ReplaceGlobalConstantsPass &other)
    : PassWrapper(other) {
    global_to_value_ = other.global_to_value_;
  }

  llvm::StringRef getArgument() const final {
    return "replace-global-constants";
  }
  llvm::StringRef getDescription() const final {
    return "Replaces jax.global_constant arguments with constant values.";
  }

  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<mlir::func::FuncOp>()) {
      runOnFunction(funcOp);
    }
  }

  void runOnFunction(mlir::func::FuncOp funcOp) {
    if (funcOp.isExternal() || funcOp.empty()) {
      return;
    }

    mlir::Block &entryBlock = funcOp.front();
    mlir::MLIRContext *context = funcOp.getContext();

    // Replace all occurences of the arguments with a constant.
    llvm::BitVector should_erase(entryBlock.getNumArguments(), false);
    for (size_t i = 0; i < entryBlock.getNumArguments(); ++i) {
      // Get the 'jax.global_constant' attribute by its name
      auto attr =
        funcOp.getArgAttrOfType<mlir::StringAttr>(i, "jax.global_constant");
      if (!attr) {
        // This argument doesn't have the 'jax.global_constant' attribute.
        continue;
      }
      std::string constantName = attr.getValue().str();

      auto it = global_to_value_.find(constantName);
      if (it == global_to_value_.end()) {
        // This global constant is not targeted for replacement.
        continue;
      }

      mlir::BlockArgument blockArg = entryBlock.getArgument(i);
      if (checkScalarType(it->second, blockArg.getType()).failed()) {
        funcOp.emitError("argument is not of correct scalar type.");
        return signalPassFailure();
      }

      // Create constant
      mlir::OpBuilder builder(context);
      mlir::Attribute valueAttr =
        createConstantAttr(it->second, &entryBlock, &builder);
      if (!valueAttr) {
        funcOp.emitError(
          "Failed to create DenseElementsAttr for the constant value.");
        return signalPassFailure();
      }
      mlir::Location loc = funcOp.getLoc();
      auto constantOp =
        builder.create<mlir::stablehlo::ConstantOp>(loc, valueAttr);

      // Replace uses
      blockArg.replaceAllUsesWith(constantOp.getResult());
    }
  }
};

struct RemoveShapeAssertionsPass
  : public mlir::PassWrapper<
      RemoveShapeAssertionsPass, mlir::OperationPass<mlir::ModuleOp>> {
 protected:
  std::string target_function_name_;

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveShapeAssertionsPass)

  RemoveShapeAssertionsPass() = default;
  RemoveShapeAssertionsPass(const RemoveShapeAssertionsPass &other) = default;

  llvm::StringRef getArgument() const final {
    return "remove-shape-assertions-pass";
  }
  llvm::StringRef getDescription() const final {
    return "Removes shape_assertion custom calls.";
  }

  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<mlir::func::FuncOp>()) {
      runOnFunction(funcOp);
    }
  }

  void runOnFunction(mlir::func::FuncOp funcOp) {
    llvm::SmallVector<mlir::stablehlo::CustomCallOp, 4> customCallsToRemove;
    funcOp.walk([&](mlir::stablehlo::CustomCallOp customCallOp) {
      if (customCallOp.getCallTargetName() == "shape_assertion") {
        customCallsToRemove.push_back(customCallOp);
      }
    });
    for (mlir::stablehlo::CustomCallOp op : customCallsToRemove) {
      if (op->getNumResults() > 0) {
        funcOp.emitError(
          "Expected stablehlo.custom_call @shape_assertion to have no "
          "results.");
        return signalPassFailure();
      }
      op.erase();
    }
  }
};

}  // namespace

std::unique_ptr<::mlir::Pass> createReplaceFuncArgWithConstantPass(
  ConstantArgumentTransformLocMap arg_to_value,
  std::vector<std::string> target_function_names) {
  return std::make_unique<ReplaceFuncArgWithConstantPass>(
    std::move(arg_to_value), std::move(target_function_names));
}

std::unique_ptr<::mlir::Pass> createReplaceGlobalConstantsPass(
  ConstantArgumentTransformMap global_to_value) {
  return std::make_unique<ReplaceGlobalConstantsPass>(
    std::move(global_to_value));
}

std::unique_ptr<::mlir::Pass> createRemoveShapeAssertionsPass() {
  return std::make_unique<RemoveShapeAssertionsPass>();
}

}  // namespace util
}  // namespace runtime
}  // namespace xla_launcher
