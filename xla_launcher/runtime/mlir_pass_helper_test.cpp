/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 */

#include "xla_launcher/runtime/mlir_pass_helper.hpp"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace xla_launcher {
namespace runtime {
namespace util {

TEST(MlirPassTest, ReplaceFuncArgWithConstant) {
  const char *mlir_module_str = R"MLIR(
    func.func @main(%arg0: tensor<f32>, %arg1: tensor<i32>, %arg2: tensor<1xf32>, %arg3: tensor<?xf32>, %arg4: tensor<*xf32>, %arg5: !stablehlo.token) {
      "test.use"(%arg0, %arg1) : (tensor<f32>, tensor<i32>) -> ()
      return
    }
  )MLIR";

  mlir::MLIRContext context;
  context.allowUnregisteredDialects();  // For "test.use"
  context
    .loadDialect<mlir::func::FuncDialect, mlir::stablehlo::StablehloDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module =
    mlir::parseSourceString<mlir::ModuleOp>(mlir_module_str, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);

  std::map<unsigned int, ConstantArgumentTransform> arg_to_value;
  arg_to_value[0] = 42.0f;                      // Replace %arg0: tensor<f32>
  arg_to_value[1] = static_cast<int32_t>(123);  // Replace %arg1: tensor<i32>

  pm.nest<mlir::func::FuncOp>().addPass(
    createReplaceFuncArgWithConstantPass(std::move(arg_to_value), "main"));

  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);

  // After replacing arg0 and arg1, there should be 4 arguments left.
  ASSERT_EQ(main_func.getNumArguments(), 4);
  ASSERT_EQ(main_func.getFunctionType().getNumInputs(), 4);

  // The new first argument should be the old third argument (%arg2).
  auto new_arg0_type =
    llvm::dyn_cast<mlir::RankedTensorType>(main_func.getArgument(0).getType());
  ASSERT_TRUE(new_arg0_type);
  EXPECT_TRUE(new_arg0_type.getElementType().isF32());
  ASSERT_EQ(new_arg0_type.getRank(), 1);
  EXPECT_EQ(new_arg0_type.getShape()[0], 1);

  // Check that two stablehlo.constant ops were created.
  int constant_count = 0;
  bool f32_found = false;
  bool i32_found = false;
  main_func.walk([&](mlir::stablehlo::ConstantOp op) {
    constant_count++;
    auto dense_attr = llvm::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    ASSERT_TRUE(dense_attr);
    if (dense_attr.getElementType().isF32()) {
      EXPECT_EQ(dense_attr.getValues<float>()[0], 42.0f);
      f32_found = true;
    } else if (dense_attr.getElementType().isInteger(32)) {
      EXPECT_EQ(dense_attr.getValues<int32_t>()[0], 123);
      i32_found = true;
    }
  });
  EXPECT_EQ(constant_count, 2);
  EXPECT_TRUE(f32_found);
  EXPECT_TRUE(i32_found);

  // Check that the operands of "test.use" have been replaced by the constants.
  int use_count = 0;
  main_func.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "test.use") {
      use_count++;
      EXPECT_TRUE(llvm::isa<mlir::stablehlo::ConstantOp>(
        op->getOperand(0).getDefiningOp()));
      EXPECT_TRUE(llvm::isa<mlir::stablehlo::ConstantOp>(
        op->getOperand(1).getDefiningOp()));
    }
  });
  EXPECT_EQ(use_count, 1);
}

}  // namespace util
}  // namespace runtime
}  // namespace xla_launcher

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
