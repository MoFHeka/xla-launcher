/*
 * @copyright
 * BSD 3-Clause License, 2025, He Jia <mofhejia@163.com>
 * BSD 3-Clause License, 2023, pytorch-tpu
 */

#include "xla_launcher/runtime/stablehlo_helper.hpp"

#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/api/PortableApi.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla_launcher/runtime/debug_macros.hpp"
#include "xla_launcher/runtime/stablehlo_composite_helper.hpp"
#include "xla_launcher/runtime/sys_util.hpp"
#include "xla_launcher/runtime/xla_mlir_debuginfo_helper.hpp"
#include "xla_launcher/runtime/xla_util.hpp"

namespace xla_launcher {
namespace runtime {
namespace util {

class ScopedDiagnosticHandler::Impl {
 public:
  explicit Impl(mlir::MLIRContext* context)
    : context_(context), diagnostics_os_(diagnostics_str_) {
    handler_id_ = context_->getDiagEngine().registerHandler(
      [&](mlir::Diagnostic& diag) { diagnostics_os_ << diag << "\n"; });
  }

  ~Impl() { context_->getDiagEngine().eraseHandler(handler_id_); }

  std::string getDiagnostics() { return diagnostics_str_; }

 private:
  mlir::MLIRContext* context_;
  mlir::DiagnosticEngine::HandlerID handler_id_;
  std::string diagnostics_str_;
  llvm::raw_string_ostream diagnostics_os_;
};

ScopedDiagnosticHandler::ScopedDiagnosticHandler(mlir::MLIRContext* context)
  : impl_(new Impl(context)) {}

ScopedDiagnosticHandler::~ScopedDiagnosticHandler() = default;

std::string ScopedDiagnosticHandler::getDiagnostics() const {
  return impl_->getDiagnostics();
}

constexpr char kMainFnName[] = "main";
constexpr char kWrapperJaxExportMainFnName[] = "_wrapped_jax_export_main";

static std::string getHloModuleStr(const xla::HloModuleProto* proto) {
  auto hlo_module = util::CreateModuleFromProto(*proto);
  return hlo_module.value()->ToString();
}

static std::string getMlirModuleStr(mlir::ModuleOp& mlir_module) {
  std::string txt_mlir_module;
  llvm::raw_string_ostream os{txt_mlir_module};
  // Enable Debug Info to include source line info in the StableHLO dump.
  mlir::OpPrintingFlags flags;
  static bool withSrcLineInfo = sys_util::GetEnvBool("XLA_HLO_DEBUG", false);
  if (withSrcLineInfo) {
    flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/true);
  }
  mlir_module.print(os, flags);
  return txt_mlir_module;
}

static std::string getMlirModuleBytecode(mlir::ModuleOp& mlir_module) {
  std::string txt_mlir_module;
  llvm::raw_string_ostream os{txt_mlir_module};
  const std::string stablehlo_version =
    mlir::vhlo::Version::getCurrentVersion().toString();
  auto result = mlir::stablehlo::serializePortableArtifact(
    mlir_module, /* target_version = */ stablehlo_version, os);
  XLA_CHECK(result.succeeded()) << "Serializing StableHLO Failed";
  return txt_mlir_module;
}

static absl::Status ConvertHloToMhlo(
  const xla::HloModuleProto* proto, mlir::ModuleOp* mlir_module) {
  auto status = xla::ConvertHloToMlirHlo(
    *mlir_module, proto,
    /*import_all_computations=*/false);
  if (!status.ok()) {
    return status;
  }
  if (!mlir::verify(*mlir_module).succeeded()) {
    return absl::Status(
      absl::StatusCode::kInternal,
      "MHLO Module from HLO -> MHLO conversion is not legal.");
  }
  return absl::OkStatus();
}

static absl::Status mhloToStablehloHelper(
  mlir::ModuleOp* mlir_module, mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  pm.addPass(CreatePrepareXlaMlirDebuginfoPass());
  // legalize `mhlo.dot` to `mhlo.dot_general` to workaround the shape
  // refinement issue in `stablehlo.dot`.
  // TODO(lsy323): Remove this pass when mhlo.dot will can be leagalized to
  // stablehlo.dot_general in MHLO->StableHLO converter. Or shape refinement
  // logic is fixed for stablehlo.dot.
  pm.addNestedPass<mlir::func::FuncOp>(
    mlir::mhlo::createLegalizeDotToDotGeneralPass());
  // Apply pass to remove HLO tuple output, as MHLO/StableHLO supports multiple
  // outputs.
  pm.addPass(mlir::mhlo::createExpandHloTuplesPass());
  // Canonicalization after tuple flatten, to remove unused tuple op.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  // Group patterns into StableHLO composites.
  pm.addPass(CreateBuildStableHLOCompositePass());
  pm.addNestedPass<mlir::func::FuncOp>(CreateRemoveXlaMarkTensorOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());

  ScopedDiagnosticHandler diag_handler(context);
  if (mlir::failed(pm.run(*mlir_module))) {
    return absl::Status(
      absl::StatusCode::kInternal,
      "StableHLO Module from MHLO -> StableHLO conversion is not legal. "
      "Error: "
        + diag_handler.getDiagnostics());
  }
  return absl::OkStatus();
}

void ConvertHloToStableHlo(
  const xla::HloModuleProto* proto, mlir::ModuleOp* mlir_module) {
  static const std::string err_msg =
    "Please open a github issue to openxla/xla.\nOriginal HLO dump:\n";
  auto status = ConvertHloToMhlo(proto, mlir_module);
  XLA_CHECK(status.ok()) << "HLO -> MHLO conversion failed.\n"
                         << status.message() << err_msg
                         << getHloModuleStr(proto);
  status = mhloToStablehloHelper(mlir_module, mlir_module->getContext());
  XLA_CHECK(status.ok()) << "MHLO -> StableHLO conversion failed.\n"
                         << status.message() << err_msg
                         << getHloModuleStr(proto);
}

void CanonicalizeStableHloStaticShape(
  mlir::ModuleOp* mlir_module, mlir::MLIRContext* context,
  const ArgumentTransformVec& transforms,
  const ConstantArgumentTransformMap& global_constants_map) {
  ScopedDiagnosticHandler diag_handler(context);

  // Parse the refined types.
  llvm::SmallVector<mlir::Type> parsed_types;
  for (const auto& transform : transforms) {
    if (std::holds_alternative<RefineType>(transform)) {
      const auto& refine_type = std::get<RefineType>(transform);
      mlir::Type parsed_type = mlir::parseType(refine_type, context);
      XLA_CHECK(parsed_type) << "Failed to parse type " << refine_type
                             << " Error: " << diag_handler.getDiagnostics();
      parsed_types.push_back(parsed_type);
    }
  }
  mlir::PassManager pm(
    context, mlir::PassManager::getAnyOpAnchorName(),
    mlir::PassManager::Nesting::Implicit);

  // Pass 1: replace constant arguments.
  ConstantArgumentTransformLocMap arg_to_value;
  for (unsigned int i = 0; i < transforms.size(); i++) {
    if (std::holds_alternative<ConstantArgumentTransform>(transforms[i])) {
      const auto& replace_with_constant =
        std::get<ConstantArgumentTransform>(transforms[i]);
      arg_to_value[i] = replace_with_constant;
    }
  }
  if (!arg_to_value.empty()) {
    pm.addPass(
      createReplaceFuncArgWithConstantPass(arg_to_value, {kMainFnName}));
  }
  // Pass 2: refine program input shapes to be static.
  if (!parsed_types.empty()) {
    pm.addPass(
      mlir::stablehlo::createStablehloRefineArgumentsPass(parsed_types));
  }
  // Pass 3: replace global jax constants.
  if (!global_constants_map.empty()) {
    pm.addPass(createReplaceGlobalConstantsPass(global_constants_map));
  }
  // Pass 4: propagates static shapes across the program.
  pm.addPass(mlir::stablehlo::createStablehloRefineShapesPass());
  // Pass 5: remove shape assertions.
  pm.addPass(createRemoveShapeAssertionsPass());
  // Pass 6: replaces dynamic shape ops with static shape ops if possible.
  pm.addNestedPass<mlir::func::FuncOp>(
    mlir::stablehlo::createStablehloCanonicalizeDynamismPass());
  // Pass 7: simplify and propagate constants.
  pm.addNestedPass<mlir::func::FuncOp>(
    mlir::stablehlo::createStablehloAggressiveSimplificationPass());

  if (mlir::failed(pm.run(*mlir_module))) {
    XLA_CHECK(false) << "Failed to canonicalize StableHLO Module from "
                        "CanonicalizeStableHloStaticShape. Error: "
                     << diag_handler.getDiagnostics();
  }
}

void CanonicalizeStableHloStaticShape(
  mlir::ModuleOp* mlir_module, mlir::MLIRContext* context,
  const ArgumentTransformLocMap& transforms_loc_map,
  const ConstantArgumentTransformMap& global_constants_map) {
  mlir::func::FuncOp main_fn =
    mlir_module->lookupSymbol<mlir::func::FuncOp>(kMainFnName);
  XLA_CHECK(main_fn) << "Failed to find " << kMainFnName << " function.";

  size_t num_args = main_fn.getNumArguments();
  std::vector<ArgumentTransform> transforms(num_args);
  for (size_t i = 0; i < num_args; ++i) {
    auto it = transforms_loc_map.find(i);
    if (it != transforms_loc_map.end()) {
      transforms[i] = it->second;
    } else {
      // Use an empty RefineType as a placeholder for no-op.
      transforms[i] = RefineArgumentTransform(RefineType(""));
    }
  }

  CanonicalizeStableHloStaticShape(
    mlir_module, context, transforms, global_constants_map);
}

void CanonicalizeStableHloStaticShape(
  mlir::ModuleOp* mlir_module, mlir::MLIRContext* context,
  const ArgumentTransformMap& transforms_map,
  const ConstantArgumentTransformMap& global_constants_map) {
  if (transforms_map.empty()) {
    CanonicalizeStableHloStaticShape(
      mlir_module, context, std::vector<ArgumentTransform>(),
      global_constants_map);
    return;
  }
  mlir::func::FuncOp main_fn =
    mlir_module->lookupSymbol<mlir::func::FuncOp>(kMainFnName);
  XLA_CHECK(main_fn) << "Failed to find " << kMainFnName << " function.";
  XLA_CHECK(main_fn.getNumArguments() == transforms_map.size())
    << "Number of arguments in " << kMainFnName << " ("
    << main_fn.getNumArguments() << ") does not match transforms_map size ("
    << transforms_map.size() << ").";

  std::vector<ArgumentTransform> transforms;
  size_t num_args = main_fn.getNumArguments();
  transforms.resize(num_args);
  for (size_t i = 0; i < num_args; ++i) {
    mlir::BlockArgument arg = main_fn.getArgument(i);
    auto loc = arg.getLoc();
    auto name_loc = mlir::dyn_cast<mlir::NameLoc>(loc);
    XLA_CHECK(name_loc)
      << "Argument " << i << " of function " << kMainFnName
      << " does not have a NameLoc from which to extract an argument name.";
    std::string arg_name = name_loc.getName().str();
    auto it = transforms_map.find(arg_name);
    XLA_CHECK(it != transforms_map.end())
      << "Argument " << arg_name << " not found in transforms_map.";
    transforms[i] = it->second;
  }

  CanonicalizeStableHloStaticShape(
    mlir_module, context, transforms, global_constants_map);
}

std::string HloToStablehlo(
  const xla::HloModuleProto* proto, bool emit_bytecode) {
  mlir::MLIRContext context;
  bool legacy_conversion =
    sys_util::GetEnvBool("XLA_LAUNCHER_LEGACY_CONVERSION", true);
  mlir::ModuleOp mlir_module;
  if (legacy_conversion) {
    mlir_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    ConvertHloToStableHlo(proto, &mlir_module);
  } else {
    static const std::string err_msg =
      "Please open a github issue to openxla/xla.\nOriginal HLO dump:\n";
    auto status_or = xla::ConvertHloToStablehlo(context, proto);
    XLA_CHECK(status_or.ok())
      << "HLO -> StableHLO conversion failed.\n"
      << status_or.status().message() << err_msg << getHloModuleStr(proto);
    mlir_module = *(status_or.value());
  }
  if (emit_bytecode) {
    return getMlirModuleBytecode(mlir_module);
  } else {
    return getMlirModuleStr(mlir_module);
  }
}

std::string GetHloModuleStr(const xla::HloModuleProto* proto) {
  auto hlo_module = util::CreateModuleFromProto(*proto);
  return hlo_module.value()->ToString();
}

static absl::Status ConvertStablehloToMhlo(
  mlir::ModuleOp* mlir_module, mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  if (!mlir::succeeded(pm.run(*mlir_module))) {
    return absl::Status(
      absl::StatusCode::kInternal,
      "StableHLO Module from StableHLO -> MHLO conversion is not leagal.");
  }
  return absl::OkStatus();
}

static absl::Status MhloToHloHelper(
  const mlir::ModuleOp* mlir_module, xla::HloProto* hlo_proto) {
  mlir::MlirToHloConversionOptions options;
  options.propagate_layouts = true;
  options.direct_stablehlo_to_hlo = true;
  auto status = mlir::ConvertMlirHloToHlo(
    *mlir_module, hlo_proto,
    /*use_tuple_args=*/false,
    /*return_tuple=*/false, options);
  if (!status.ok()) {
    return status;
  }
  return absl::OkStatus();
}

void ConvertStableHloToHlo(
  mlir::ModuleOp* mlir_module, mlir::MLIRContext* context,
  xla::HloProto* hlo_proto) {
  static const std::string err_msg =
    "Please open a github issue to openxla/xla.\nOriginal StableHLO dump:\n";
  bool legacy_conversion =
    sys_util::GetEnvBool("XLA_LAUNCHER_LEGACY_CONVERSION", true);
  if (legacy_conversion) {
    auto status = ConvertStablehloToMhlo(mlir_module, context);
    XLA_CHECK(status.ok()) << "StableHLO -> MHLO conversion failed.\n"
                           << status.message() << err_msg
                           << getMlirModuleStr(*mlir_module);
    status = MhloToHloHelper(mlir_module, hlo_proto);
    XLA_CHECK(status.ok()) << "MHLO -> XLA HLO conversion failed.\n"
                           << status.message() << err_msg
                           << getMlirModuleStr(*mlir_module);
  } else {
    auto status = xla::ConvertStablehloToHloProto(*mlir_module, hlo_proto);
    XLA_CHECK(status.ok()) << "StableHLO -> XLA HLO conversion failed.\n"
                           << status.message() << err_msg
                           << getMlirModuleStr(*mlir_module);
  }
}

const std::string GetTorchDtypeToStablehloDtype(const std::string& dtype) {
  if (dtype == "int8") return "i8";
  if (dtype == "uint8") return "ui8";
  if (dtype == "int16") return "i16";
  if (dtype == "int32") return "i32";
  if (dtype == "int64") return "i64";
  XLA_ERROR() << "Unsupported dtype for conversion to Stablehlo type: "
              << dtype;
}

const std::unordered_map<xla::PrimitiveType, std::string>&
GetHloDtypeToStablehloDtypeMap() {
  static const std::unordered_map<xla::PrimitiveType, std::string> m_{
    {xla::PrimitiveType::S4, "i4"},    {xla::PrimitiveType::S8, "i8"},
    {xla::PrimitiveType::S16, "i16"},  {xla::PrimitiveType::S32, "i32"},
    {xla::PrimitiveType::S64, "i64"},  {xla::PrimitiveType::U4, "ui4"},
    {xla::PrimitiveType::U8, "ui8"},   {xla::PrimitiveType::U16, "ui16"},
    {xla::PrimitiveType::U32, "ui32"}, {xla::PrimitiveType::U64, "ui64"},
    {xla::PrimitiveType::F16, "f16"},  {xla::PrimitiveType::BF16, "bf16"},
    {xla::PrimitiveType::F32, "f32"},  {xla::PrimitiveType::F64, "f64"},
  };
  return m_;
}

std::optional<xla::PrimitiveType> GetIntDtypeToHloDtype(
  const std::string& dtype) {
  if (dtype == "int8") return xla::PrimitiveType::S8;
  if (dtype == "uint8") return xla::PrimitiveType::U8;
  if (dtype == "int16") return xla::PrimitiveType::S16;
  if (dtype == "int32") return xla::PrimitiveType::S32;
  if (dtype == "int64") return xla::PrimitiveType::S64;
  XLA_ERROR() << "Unsupported dtype for conversion to Hlo type: " << dtype;
  return std::nullopt;
}

}  // namespace util
}  // namespace runtime
}  // namespace xla_launcher
