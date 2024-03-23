#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "Conversions/Tosa/Passes.h"
#include "HOM/HOMOps.h"
#include "WeightsEngine/WeightsEngine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"

#define PASS_NAME "hom-serialize-weight"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DEF_HOMSERIALIZEWEIGHTPASS
#include "HOM/Passes.h.inc"

namespace {

struct SerializeTosaConstOp : public OpRewritePattern<tosa::ConstOp> {
  using OpRewritePattern<tosa::ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ConstOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = op.getValueAttr();

    auto idx = gWe.addWeight(value);

    auto constantOP = rewriter.create<ConstantOp>(loc, value.getType(), idx);

    while (!op->getUses().empty()) {
      op->getUses().begin()->set(constantOP.getResult());
    }

    rewriter.eraseOp(op);

    return success();
  }
};

struct HOMSerializeWeightPass
    : impl::HOMSerializeWeightPassBase<HOMSerializeWeightPass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

LogicalResult HOMSerializeWeightPass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);

  patternList.add<SerializeTosaConstOp>(ctx);
  patterns = std::move(patternList);
  return success();
}

void HOMSerializeWeightPass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

} // namespace
} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir
