#include "Conversions/HOM/Passes.h"
#include "Conversions/Tosa/Passes.h"
#include "HOM/HOMOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define PASS_NAME "hom-to-homnvgpu"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {

#define GEN_PASS_DEF_HOMTOHOMNVGPUPASS
#include "Conversions/HOM/Passes.h.inc"

#include "Conversions/HOM/HOMToHOMNVGPU.pdll.h.inc"

namespace {
struct HOMToHOMNVGPUPass : impl::HOMToHOMNVGPUPassBase<HOMToHOMNVGPUPass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

LogicalResult HOMToHOMNVGPUPass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);

  populateGeneratedPDLLPatterns(patternList);
  patterns = std::move(patternList);
  return success();
}

void HOMToHOMNVGPUPass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}
} // namespace

} // namespace hands_on_mlir
} // namespace mlir
