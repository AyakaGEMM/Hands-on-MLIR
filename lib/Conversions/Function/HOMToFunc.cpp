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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "Conversions/Function/FunctionCallUtils.h"
#include "Conversions/Function/Passes.h"
#include "HOM/HOMOps.h"
#include "WeightsEngine/WeightsEngine.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

#define PASS_NAME "hom-to-func"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DEF_HOMTOFUNCPASS
#include "Conversions/Function/Passes.h.inc"

namespace {

static void maybeInsertDeallocFn(OpBuilder &builder, Operation *op,
                                 ArrayRef<Value> operands) {
  assert(operands.size() == 1);

  auto funcOp = op->getParentOfType<func::FuncOp>();

  for (const auto &user : operands[0].getUsers()) {
    if (dyn_cast<func::ReturnOp>(user)) {
      return;
    }
  }

  auto moduleOp = op->getParentOfType<ModuleOp>();
  auto deallocFn = lookupOrCreateDeallocF32Fn(moduleOp);
  static std::optional<OpBuilder::InsertPoint> p;
  OpBuilder::InsertionGuard g(builder);
  if (p == std::nullopt) {
    func::ReturnOp returnOp;
    funcOp->walk([&](func::ReturnOp op) {
      returnOp = op;
    }); // Really stupid way to get the return op with O(N) complexity.
    builder.setInsertionPoint(returnOp);
    p = builder.saveInsertionPoint();
  } else {
    builder.restoreInsertionPoint(p.value());
  }
  builder.create<func::CallOp>(funcOp.getLoc(), deallocFn, operands);
}

struct HOMToFuncPass : impl::HOMToFuncPassBase<HOMToFuncPass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

struct ConvertHOMMatmulOp : public OpConversionPattern<MatmulAddOp> {
  using OpConversionPattern<MatmulAddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MatmulAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto returnType = dyn_cast<TensorType>(op.getOutput().getType());
    func::FuncOp allocFn;

    if (returnType.getElementType().isF32()) {
      allocFn = lookupOrCreateAlloc3DMemRefF32Fn(moduleOp);
    } else {
      llvm_unreachable("Not good.");
    }

    // Stupid Static Shape Inference Here. Should convert to dynamic shape if I
    // have time.
    auto A = rewriter.create<arith::ConstantIntOp>(
        loc, returnType.getShape()[0], 32);
    auto B = rewriter.create<arith::ConstantIntOp>(
        loc, returnType.getShape()[1], 32);
    auto C = rewriter.create<arith::ConstantIntOp>(
        loc, returnType.getShape()[2], 32);

    SmallVector<Value> allocOperands = {A.getResult(), B.getResult(),
                                        C.getResult()};
    auto allocCaller =
        rewriter.create<func::CallOp>(loc, allocFn, allocOperands);

    auto funcOp = lookupOrCreateMatmulAddF32Fn(moduleOp);

    SmallVector<Value> operands = {op.getOperand0(), op.getOperand1(),
                                   op.getOperand2(), allocCaller->getResult(0)};
    rewriter.create<func::CallOp>(op.getLoc(), funcOp, operands);

    while (!op->getUses().empty()) {
      op->getUses().begin()->set(allocCaller->getResult(0));
    }

    maybeInsertDeallocFn(rewriter, op, {allocCaller->getResult(0)});
    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertHOMConstantOp : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto allocFn = lookupOrCreateAllocConstantF32Fn(moduleOp);

    auto idx = rewriter.create<arith::ConstantIntOp>(
        loc, op.getIdxAttr().getInt(), 32);

    SmallVector<Value> operands = {idx->getResult(0)};
    auto allocCaller = rewriter.create<func::CallOp>(loc, allocFn, operands);

    while (!op.use_empty()) {
      op->getUses().begin()->set(allocCaller->getResult(0));
    }

    maybeInsertDeallocFn(rewriter, op, {allocCaller->getResult(0)});
    rewriter.eraseOp(op);

    return success();
  }
};

LogicalResult HOMToFuncPass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);
  patterns = std::move(patternList);
  return success();
}

void HOMToFuncPass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);

  auto *context = &getContext();
  RewritePatternSet convPatterns(context);
  ConversionTarget target(*context);

  // To-do: Wrap it to a custom TypeConverter constructor.
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([](RankedTensorType type) {
    return UnrankedMemRefType::get(type.getElementType(), 0);
  });
  typeConverter.addConversion([](UnrankedTensorType type) -> Type {
    return UnrankedMemRefType::get(type.getElementType(), 0);
  });

  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(convPatterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(convPatterns, typeConverter);
  populateReturnOpTypeConversionPattern(convPatterns, typeConverter);

  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });

  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });

  convPatterns.add<ConvertHOMMatmulOp>(typeConverter, context);
  convPatterns.add<ConvertHOMConstantOp>(typeConverter, context);

  target.addLegalDialect<func::FuncDialect, arith::ArithDialect>();
  target.addIllegalDialect<hom::HOMDialect>();

  if (failed(applyFullConversion(getOperation(), target,
                                 std::move(convPatterns)))) {
    signalPassFailure();
  }
}

} // namespace
} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir
