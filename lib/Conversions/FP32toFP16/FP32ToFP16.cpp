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

#include "Conversions/Tosa/Passes.h"
#include "HOM/HOMOps.h"
#include "WeightsEngine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"
#include <iostream>

#define PASS_NAME "hom-fp32-to-fp16"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {

#define GEN_PASS_DEF_HOMFP32TOFP16PASS
#include "Conversions/FP32toFP16/Passes.h.inc"

#include "Conversions/FP32toFP16/HOMFP32ToFP16.pdll.h.inc"

namespace {

static void generateCastOpImpl(PatternRewriter &rewriter, Operation *op0) {
  auto constOp = dyn_cast<tosa::ConstOp>(op0);
  tosa::ConstOp newOp;
  auto fn = [&]<typename T>(std::shared_ptr<T> dataPtr) {
    newOp = hom::foldCast<T>(rewriter, constOp, rewriter.getF16Type(),
                             dataPtr.get(),
                             constOp.getOutput().getType().getShape());
  };

  universalCastElementsToPtr(constOp.getValue(), fn);

  constOp.replaceAllUsesWith(newOp.getResult());
}

struct HOMFP32ToFP16Pass : impl::HOMFP32ToFP16PassBase<HOMFP32ToFP16Pass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

struct ConvertHOMOpType : public RewritePattern {

public:
  ConvertHOMOpType(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    bool isModified = false;

    if (dyn_cast<hom::HOMDialect>(op->getDialect()) ||
        dyn_cast<tosa::TransposeOp>(op)) {
      for (auto res : op->getResults()) {
        if (auto tp = dyn_cast<RankedTensorType>(res.getType())) {
          if (tp.getElementType().isF32()) {
            auto newTp =
                RankedTensorType::get(tp.getShape(), rewriter.getF16Type());
            res.setType(newTp);
            isModified = true;
          }
        }
      }
      return success(isModified);
    }

    return failure();
  }
};

LogicalResult HOMFP32ToFP16Pass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);

  populateGeneratedPDLLPatterns(patternList);
  patternList.getPDLPatterns().registerRewriteFunction("generateCastOp",
                                                       generateCastOpImpl);
  patternList.add<ConvertHOMOpType>(ctx);
  patterns = std::move(patternList);
  return success();
}

void HOMFP32ToFP16Pass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);

  auto *context = &getContext();
  RewritePatternSet convPatterns(context);
  ConversionTarget target(*context);

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([context](RankedTensorType type) {
    auto elementTp = type.getElementType();
    if (elementTp.isF32()) {
      return RankedTensorType::get(type.getShape(), Float16Type::get(context));
    }
    return type;
  });
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(convPatterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(convPatterns, typeConverter);
  populateReturnOpTypeConversionPattern(convPatterns, typeConverter);

  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });

  target
      .addLegalDialect<func::FuncDialect, tosa::TosaDialect, hom::HOMDialect>();

  if (failed(applyFullConversion(getOperation(), target,
                                 std::move(convPatterns)))) {
    signalPassFailure();
  }
}

} // namespace
} // namespace hands_on_mlir
} // namespace mlir
