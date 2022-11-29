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

#include "CpuGemm/CGDialect.h"
#include "CpuGemm/CGOps.h"

namespace mlir
{
    namespace hands_on_mlir
    {
        void registerConvVectorizationPass();
        void registerPointwiseConvToGemmPass();
        void registerPoolingVectorizationPass();
        void registerLowerBudPass();
        void registerLowerDIPPass();
        void registerLowerDAPPass();
        void registerLowerRVVPass();
        void registerMatMulOptimizePass();
        void registerConvOptimizePass();
        void registerLowerVectorExpPass();
    } // namespace hands_on_mlir
} // namespace mlir

int main(int argc, char **argv)
{
    // Register all MLIR passes.
    mlir::registerAllPasses();
    mlir::buddy::registerPointwiseConvToGemmPass();
    // Register Vectorization of Convolution.
    mlir::buddy::registerConvVectorizationPass();
    // Register Vectorization of Pooling.
    mlir::buddy::registerPoolingVectorizationPass();
    mlir::buddy::registerLowerBudPass();
    mlir::buddy::registerLowerDIPPass();
    mlir::buddy::registerLowerDAPPass();
    mlir::buddy::registerLowerRVVPass();
    mlir::buddy::registerLowerVectorExpPass();

    // Register Several Optimize Pass.
    mlir::buddy::registerMatMulOptimizePass();
    mlir::buddy::registerConvOptimizePass();

    mlir::DialectRegistry registry;
    // Register all MLIR core dialects.
    registerAllDialects(registry);
    // Register dialects in buddy-mlir project.
    // clang-format off
  registry.insert<buddy::bud::BudDialect,
                  buddy::dip::DIPDialect,
                  buddy::dap::DAPDialect,
                  buddy::rvv::RVVDialect,
                  buddy::vector_exp::VectorExpDialect>();
    // clang-format on

    return mlir::failed(
        mlir::MlirOptMain(argc, argv, "buddy-mlir optimizer driver", registry));
}