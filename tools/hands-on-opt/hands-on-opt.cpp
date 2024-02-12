#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "HOM/HOMOps.h"
#include "HOM/Passes.h"
#include "InitAllPasses.h"

int main(int argc, char **argv) {
  // Register all MLIR passes.
  mlir::registerAllPasses();
  mlir::hands_on_mlir::registerAllHOMPasses();

  mlir::DialectRegistry registry;
  // Register all MLIR core dialects.
  registerAllDialects(registry);
  // Register dialects in buddy-mlir project.
  // clang-format off
  registry.insert<mlir::hands_on_mlir::hom::HOMDialect>();

  // clang-format on

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "hands-on-mlir optimizer driver", registry));
}
