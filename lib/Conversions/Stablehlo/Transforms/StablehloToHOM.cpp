/* Copyright 2022 OpenXLA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "Conversions/Stablehlo/Passes.h"
#include "HOM/HOMOps.h"
#include "WeightsEngine/WeightsEngine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"

#define PASS_NAME "stablehlo-to-hom"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DEF_STABLEHLOTOHOMPASS
#include "Conversions/Stablehlo/Passes.h.inc"
#include "Conversions/Stablehlo/StablehloToHOM.pdll.h.inc"

namespace {

struct StablehloToHOMPass : impl::StablehloToHOMPassBase<StablehloToHOMPass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

struct ConvertStablehloConstantOp
    : public OpRewritePattern<stablehlo::ConstantOp> {
  using OpRewritePattern<stablehlo::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto value = op.getValueAttr();

    size_t idx = 0;

    auto constantOP = rewriter.create<ConstantOp>(loc, value.getType(), idx);

    while (!op->getUses().empty()) {
      op->getUses().begin()->set(constantOP->getResult(0));
    }

    rewriter.eraseOp(op);

    return success();
  }
};

LogicalResult StablehloToHOMPass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);
  populateGeneratedPDLLPatterns(patternList);
  patternList.add<ConvertStablehloConstantOp>(ctx);
  patterns = std::move(patternList);
  return success();
}

void StablehloToHOMPass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

} // namespace
} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir
