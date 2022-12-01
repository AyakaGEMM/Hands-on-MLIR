#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "CpuGemm/IR/CGOps.h"

namespace mlir {
namespace hands_on_mlir {
void registerMatMulCPUOptimizePass();
} // namespace hands_on_mlir
} // namespace mlir

int main(int argc, char **argv) {
  // Register all MLIR passes.
  mlir::registerAllPasses();

  // Register Several Optimize Pass.
  mlir::hands_on_mlir::registerMatMulCPUOptimizePass();

  mlir::DialectRegistry registry;
  // Register all MLIR core dialects.
  registerAllDialects(registry);
  // Register dialects in buddy-mlir project.
  // clang-format off
  registry.insert<hands_on_mlir::cg::CGDialect>();
  // clang-format on

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "hands-on-mlir optimizer driver", registry));
}