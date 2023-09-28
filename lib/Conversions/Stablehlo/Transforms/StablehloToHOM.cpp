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

#include <memory>
#include <string>
#include <utility>

#include "Conversions/Stablehlo/Transforms/Passes.h"
#include "HOM/HOMOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"

#define PASS_NAME "stablehlo-to-hom"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DEF_STABLEHLOTOHOMPASS
#include "Conversions/Stablehlo/Transforms/Passes.h.inc"
#include "Conversions/Stablehlo/Transforms/StablehloToHOM.pdll.h.inc"

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

    auto shapeType = value.getShapedType();

    std::string fileName;

    if (shapeType.getElementType().isF32()) {
      static int fp32Idx = 0;
      fileName = "/home/pzzzzz/MyProjects/Hands-on-MLIR/examples/"
                 "contants2memref/" +
                 std::to_string(fp32Idx++) + ".txt";
      serializeWeightToDisk<APFloat>(value, fileName);
    }

    auto constantOP =
        rewriter.create<ConstantOp>(loc, value.getType(), fileName);

    while (!op->getUses().empty()) {
      op->getUses().begin()->set(constantOP->getResult(0));
    }

    rewriter.eraseOp(op);

    return success();
  }

  template <class T>
  static void serializeWeightToDisk(ElementsAttr &value,
                                    const std::string &fileName) {
    auto shape = value.getShapedType();
    auto data = value.getValues<T>();
    auto dimSize = shape.getShape();
    // std::ofstream file(fileName);
    std::error_code EC;
    llvm::raw_fd_ostream out(fileName, EC);
    for (auto i : dimSize) {
      out << i << " ";
    }
    out << "\n";
    auto totalSize = value.getNumElements();
    for (int i = 0; i < totalSize; i++) {
      data[i].print(out);
    }
    out << "\n";
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
