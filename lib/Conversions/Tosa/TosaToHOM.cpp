#include <cstddef>
#include <memory>
#include <string>
#include <utility>

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

#define PASS_NAME "tosa-to-hom"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DEF_TOSATOHOMPASS
#include "Conversions/Tosa/Passes.h.inc"
#include "Conversions/Tosa/TosaToHOM.pdll.h.inc"

namespace {

struct TosaToHOMPass : impl::TosaToHOMPassBase<TosaToHOMPass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

struct ConvertTosaMatmulOp : public OpRewritePattern<tosa::MatMulOp> {
  using OpRewritePattern<tosa::MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    int useSize = 0;
    for (auto iter = op->getUsers().begin(); iter != op->getUsers().end();
         ++iter, ++useSize) {
      if (useSize == 2) {
        break;
      }
    }

    if (useSize == 1) {
      if (auto addOp = llvm::dyn_cast<tosa::AddOp>(*(op->getUsers().begin()))) {
        auto homMatmulAddOp = rewriter.create<MatmulAddOp>(
            loc, addOp.getResult().getType(), op.getA(), op.getB(),
            addOp.getInput2());
        while (!addOp->getUses().empty()) {
          addOp->getUses().begin()->set(homMatmulAddOp.getResult());
        }
        rewriter.eraseOp(addOp);
        op->dropAllUses();
        rewriter.eraseOp(op);
        return success();
      }
    }

    auto homMatmulOp = rewriter.create<MatmulOp>(
        loc, op.getResult().getType(), op->getOperand(0), op->getOperand(1));

    rewriter.replaceOp(op, homMatmulOp);

    return success();
  }
};

LogicalResult TosaToHOMPass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);

  populateGeneratedPDLLPatterns(patternList);
  patternList.add<ConvertTosaMatmulOp>(ctx);
  patterns = std::move(patternList);
  return success();
}

void TosaToHOMPass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

} // namespace
} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir
