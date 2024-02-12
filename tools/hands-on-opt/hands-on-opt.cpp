#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "InitAllDialects.h"
#include "InitAllPasses.h"

int main(int argc, char **argv) {
  // Register all MLIR passes.
  mlir::registerAllPasses();
  mlir::hands_on_mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  // Register all MLIR core dialects.
  registerAllDialects(registry);
  // Register dialects in hands-on-mlir project.
  mlir::hands_on_mlir::registerAllPasses();

  // clang-format on

  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "hands-on-mlir optimizer driver", registry));
}
