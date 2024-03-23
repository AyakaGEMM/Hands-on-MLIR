#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "Conversions/Tosa/Passes.h"
#include "HOM/HOMOps.h"
#include "HOMNVGPU/HOMNVGPUOps.h"
#include "WeightsEngine/WeightsEngine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define PASS_NAME "homnvgpu-fusion"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu {

#define GEN_PASS_DEF_HOMNVGPUFUSIONPASS
#include "Dialect/HOMNVGPU/HOMNVGPUFusion.pdll.h.inc"
#include "HOMNVGPU/Passes.h.inc"

namespace {

static void generateGemmLnGemmImpl(PatternRewriter &rewriter, Operation *gemm0_,
                                   Operation *ln_, Operation *gemm1_) {

  auto gemm0 = dyn_cast<homnvgpu::MatmulOp>(gemm0_);
  auto ln = dyn_cast<homnvgpu::LayernormOp>(ln_);
  auto gemm1 = dyn_cast<homnvgpu::MatmulOp>(gemm1_);

  auto gemmWithVarMean = rewriter.create<homnvgpu::MatmulWithVarMeanOp>(
      gemm0->getLoc(), gemm0.getOperand0(), gemm0.getOperand1(),
      gemm0.getOperand2(), gemm0.getAlpha(), gemm0.getBeta(), gemm0.getAct(),
      ln.getEps());
  auto LnGemm = rewriter.create<homnvgpu::LayernormMatmulOp>(
      gemm1->getLoc(), gemm1.getResult().getType(), gemmWithVarMean.getOutput(),
      gemm1.getOperand1(), gemm1.getOperand2(), gemmWithVarMean.getVar(),
      gemmWithVarMean.getMean(), gemm1.getAlpha(), gemm1.getBeta(),
      gemm1.getAct());

  gemm1.replaceAllUsesWith(LnGemm.getResult());
}

static void updateMaskWithCuSeqLenImpl(PatternRewriter &rewriter,
                                       Operation *mask_, Operation *bert_mha_) {

  auto mask = dyn_cast<hom::MaskOp>(mask_);
  auto bert_mha = dyn_cast<homnvgpu::BertMhaOp>(bert_mha_);

  homnvgpu::CuSeqLenOp newMask;

  mask->getBlock()->walk([&](homnvgpu::CuSeqLenOp op) {
    if (op.getInput() == mask.getInput()) {
      newMask = op;
    }
  });

  if (!newMask) {
    auto shape = mask.getType().getShape();
    rewriter.setInsertionPointToStart(mask->getBlock());
    newMask = rewriter.create<homnvgpu::CuSeqLenOp>(
        mask->getLoc(),
        RankedTensorType::get({shape[0] + 1}, rewriter.getI32Type()),
        mask.getInput());
  }

  bert_mha->setOperand(1, newMask.getOutput());
}

struct HOMNVGPUFusionPass : impl::HOMNVGPUFusionPassBase<HOMNVGPUFusionPass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

LogicalResult HOMNVGPUFusionPass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);

  populateGeneratedPDLLPatterns(patternList);
  patternList.getPDLPatterns().registerRewriteFunction("generateGemmLnGemm",
                                                       generateGemmLnGemmImpl);
  patternList.getPDLPatterns().registerRewriteFunction(
      "updateMaskWithCuSeqLen", updateMaskWithCuSeqLenImpl);
  patterns = std::move(patternList);
  return success();
}

void HOMNVGPUFusionPass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

} // namespace
} // namespace homnvgpu
} // namespace hands_on_mlir
} // namespace mlir
