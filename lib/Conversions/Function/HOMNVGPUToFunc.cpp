#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "Conversions/Function/FunctionUtils.h"
#include "Conversions/Function/Passes.h"
#include "HOM/HOMOps.h"
#include "HOMNVGPU/HOMNVGPUOps.h"
#include "WeightsEngine/WeightsEngine.h"
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
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

#define PASS_NAME "homnvgpu-to-func"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {

#define GEN_PASS_DEF_HOMNVGPUTOFUNCPASS
#include "Conversions/Function/Passes.h.inc"

namespace {

struct HOMNVGPUToFuncPass : impl::HOMNVGPUToFuncPassBase<HOMNVGPUToFuncPass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

struct ConvertHOMNVGPUMatmulOp
    : public OpConversionPattern<homnvgpu::MatmulOp> {
  using OpConversionPattern<homnvgpu::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(homnvgpu::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto returnType = op.getOutput().getType();
    Value c = op.getOperand(2), d;
    auto cDefiningOp = c.getDefiningOp();

    auto canReuseC = [&]() {
      if (cDefiningOp && dyn_cast<hom::DummyTensorOp>(cDefiningOp)) {
        return false;
      }
      if (op.getBeta().convertToFloat() == 0) {
        return true;
      }
      bool hasOtherUser = false;
      bool atThisOp = true;

      op->getBlock()->walk(
          [&op, &atThisOp, &hasOtherUser, &c](Operation *visit) {
            if (atThisOp) {
              return;
            }
            if (visit == op) {
              atThisOp = false;
              return;
            }

            for (const auto &operand : visit->getOperands()) {
              if (operand == c) {
                hasOtherUser = true;
              }
            }
          });

      return !hasOtherUser;
    };

    if (canReuseC()) {
      d = c;
    } else {
      func::FuncOp allocFn;

      if (returnType.getElementType().isF32()) {
        allocFn = lookupOrCreateAlloc3DMemRefNVGPUF32Fn(moduleOp);
      } else if (returnType.getElementType().isF16()) {
        allocFn = lookupOrCreateAlloc3DMemRefNVGPUF16Fn(moduleOp);
      } else {
        llvm_unreachable("Not good.");
      }

      // To-do: Stupid Static Shape Inference Here. Should convert to dynamic
      // shape if I have time.
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

      d = allocCaller.getResult(0);

      if (cDefiningOp && dyn_cast<hom::DummyTensorOp>(cDefiningOp)) {
        c = d;
      }
    }

    func::FuncOp funcOp;

    if (returnType.getElementType().isF32()) {
      funcOp = lookupOrCreateGemmNVGPUF32Fn(moduleOp);
    } else if (returnType.getElementType().isF16()) {
      funcOp = lookupOrCreateGemmNVGPUF16Fn(moduleOp);
    } else {
      llvm_unreachable("Not good.");
    }

    auto alpha = rewriter.create<arith::ConstantFloatOp>(loc, op.getAlpha(),
                                                         rewriter.getF32Type());
    auto beta = rewriter.create<arith::ConstantFloatOp>(loc, op.getBeta(),
                                                        rewriter.getF32Type());
    auto act = rewriter.create<arith::ConstantIntOp>(loc, op.getAct(),
                                                     rewriter.getI64Type());

    auto transA = rewriter.create<arith::ConstantIntOp>(loc, op.getTransa(),
                                                        rewriter.getI1Type());
    auto transB = rewriter.create<arith::ConstantIntOp>(loc, op.getTransb(),
                                                        rewriter.getI1Type());

    SmallVector<Value> operands = {op.getOperand(0),
                                   transA.getResult(),
                                   op.getOperand(1),
                                   transB.getResult(),
                                   c,
                                   d,
                                   act.getResult(),
                                   alpha.getResult(),
                                   beta.getResult()};
    rewriter.create<func::CallOp>(op.getLoc(), funcOp, operands);

    while (!op->getUses().empty()) {
      op->getUses().begin()->set(d);
    }

    d.setType(UnrankedMemRefType::get(returnType.getElementType(), 0));

    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertHOMConstantOp : public OpConversionPattern<hom::ConstantOp> {
  using OpConversionPattern<hom::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hom::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto returnType = op.getOutput().getType();

    func::FuncOp allocFn;

    if (returnType.getElementType().isF32()) {
      allocFn = lookupOrCreateAllocConstantNVGPUF32Fn(moduleOp);
    } else if (returnType.getElementType().isF16()) {
      allocFn = lookupOrCreateAllocConstantNVGPUF16Fn(moduleOp);
    } else if (returnType.getElementType().isInteger(32)) {
      allocFn = lookupOrCreateAllocConstantNVGPUI32Fn(moduleOp);
    } else {
      returnType.dump();
      llvm_unreachable("Not good.");
    }

    auto idx = rewriter.create<arith::ConstantIntOp>(
        loc, op.getIdxAttr().getInt(), 32);

    SmallVector<Value> operands = {idx->getResult(0)};
    auto allocCaller = rewriter.create<func::CallOp>(loc, allocFn, operands);

    while (!op.use_empty()) {
      op->getUses().begin()->set(allocCaller->getResult(0));
    }

    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertHOMNVGPULayernormOp
    : public OpConversionPattern<homnvgpu::LayernormOp> {
  using OpConversionPattern<homnvgpu::LayernormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(homnvgpu::LayernormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    Type elementType;

    if (auto tensorTy = dyn_cast<TensorType>(op->getOperand(0).getType())) {
      elementType = tensorTy.getElementType();
    } else if (auto memrefTy =
                   dyn_cast<UnrankedMemRefType>(op->getOperand(0).getType())) {
      elementType = memrefTy.getElementType();
    } else {
      op->getOperand(0).getType().dump();
      llvm_unreachable("Not ok.");
    }

    func::FuncOp lnFn;

    if (elementType.isF32()) {
      lnFn = lookupOrCreateLayernormNVGPUF32Fn(moduleOp);
    } else if (elementType.isF16()) {
      lnFn = lookupOrCreateLayernormNVGPUF16Fn(moduleOp);
    } else {
      llvm_unreachable("Not ok");
    }

    auto eps = rewriter.create<arith::ConstantFloatOp>(loc, op.getEps(),
                                                       rewriter.getF32Type());

    SmallVector<Value> operands = {op.getOperand(), eps->getResult(0)};
    auto lnCaller = rewriter.create<func::CallOp>(loc, lnFn, operands);

    while (!op.use_empty()) {
      op->getUses().begin()->set(lnCaller.getOperand(0));
    }

    rewriter.eraseOp(op);

    lnCaller->getOperand(0).setType(UnrankedMemRefType::get(elementType, 0));

    return success();
  }
};

struct ConvertHOMNVGPUCuSeqLenOp
    : public OpConversionPattern<homnvgpu::CuSeqLenOp> {
  using OpConversionPattern<homnvgpu::CuSeqLenOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(homnvgpu::CuSeqLenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto returnType = op.getOutput().getType();

    Type inputElementType;

    if (auto tensorTy = dyn_cast<TensorType>(op->getOperand(0).getType())) {
      inputElementType = tensorTy.getElementType();
    } else if (auto memrefTy =
                   dyn_cast<UnrankedMemRefType>(op->getOperand(0).getType())) {
      inputElementType = memrefTy.getElementType();
    } else {
      op->getOperand(0).getType().dump();
      llvm_unreachable("Not ok.");
    }

    auto allocFn = lookupOrCreateAlloc1DMemRefNVGPUI32Fn(moduleOp);

    func::FuncOp cuSeqLenFn;

    if (inputElementType.isInteger(64)) {
      cuSeqLenFn = lookupOrCreateCuSeqLenNVGPUI64Fn(moduleOp);
    } else if (inputElementType.isInteger(32)) {
      cuSeqLenFn = lookupOrCreateCuSeqLenNVGPUI32Fn(moduleOp);
    } else {
      llvm_unreachable("Not good.");
    }

    // To-do: Stupid Static Shape Inference Here. Should convert to dynamic
    // shape if I have time.
    auto A = rewriter.create<arith::ConstantIntOp>(
        loc, returnType.getShape()[0], 32);

    SmallVector<Value> allocOperands = {A.getResult()};
    auto allocCaller =
        rewriter.create<func::CallOp>(loc, allocFn, allocOperands);

    SmallVector<Value> operands = {op.getOperand(), allocCaller->getResult(0)};
    auto cuSeqLenCaller =
        rewriter.create<func::CallOp>(loc, cuSeqLenFn, operands);

    while (!op.use_empty()) {
      op->getUses().begin()->set(cuSeqLenCaller.getOperand(1));
    }

    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertHOMNVGPUAddOp : public OpConversionPattern<homnvgpu::AddOp> {
  using OpConversionPattern<homnvgpu::AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(homnvgpu::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto returnType = op.getOutput().getType();

    func::FuncOp allocFn;
    if (returnType.getElementType().isF32()) {
      allocFn = lookupOrCreateAlloc3DMemRefNVGPUF32Fn(moduleOp);
    } else if (returnType.getElementType().isF16()) {
      allocFn = lookupOrCreateAlloc3DMemRefNVGPUF16Fn(moduleOp);
    } else {
      llvm_unreachable("Not implemented.");
    }

    // To-do: Stupid Static Shape Inference Here. Should convert to dynamic
    // shape if I have time.
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

    SmallVector<Value> operands = {op.getOperand(0), op.getOperand(1),
                                   allocCaller->getResult(0)};

    func::FuncOp addFn;

    if (returnType.getElementType().isF32()) {
      addFn = lookupOrCreateAddNVGPUF32Fn(moduleOp);
    } else if (returnType.getElementType().isF16()) {
      addFn = lookupOrCreateAddNVGPUF16Fn(moduleOp);
    } else {
      llvm_unreachable("Not implemented.");
    }

    auto addCaller = rewriter.create<func::CallOp>(loc, addFn, operands);

    while (!op.use_empty()) {
      op->getUses().begin()->set(addCaller.getOperand(2));
    }

    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertHOMNVGPUGatherOp
    : public OpConversionPattern<homnvgpu::GatherOp> {
  using OpConversionPattern<homnvgpu::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(homnvgpu::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto returnType = op.getOutput().getType();

    func::FuncOp allocFn;

    if (returnType.getElementType().isF16()) {
      allocFn = lookupOrCreateAlloc3DMemRefNVGPUF16Fn(moduleOp);
    } else {
      llvm_unreachable("Not implemented.");
    }

    // To-do: Stupid Static Shape Inference Here. Should convert to dynamic
    // shape if I have time.
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

    SmallVector<Value> operands = {op.getOperand(0), op.getOperand(1),
                                   allocCaller->getResult(0)};

    func::FuncOp gatherFn;

    if (returnType.getElementType().isF16()) {
      gatherFn = lookupOrCreateGatherNVGPUF16Fn(moduleOp);
    } else {
      llvm_unreachable("Not implemented.");
    }

    auto addCaller = rewriter.create<func::CallOp>(loc, gatherFn, operands);

    while (!op.use_empty()) {
      op->getUses().begin()->set(addCaller.getOperand(2));
    }

    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertHOMNVGPUBertMhaOp
    : public OpConversionPattern<homnvgpu::BertMhaOp> {
  using OpConversionPattern<homnvgpu::BertMhaOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(homnvgpu::BertMhaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto returnType = op.getOutput().getType();

    Type elementType;

    if (auto tensorTy = dyn_cast<TensorType>(op->getResult(0).getType())) {
      elementType = tensorTy.getElementType();
    } else if (auto memrefTy =
                   dyn_cast<MemRefType>(op->getResult(0).getType())) {
      elementType = memrefTy.getElementType();
    } else {
      llvm_unreachable("Not ok.");
    }

    func::FuncOp attnFn;

    if (elementType.isF32()) {
      attnFn = lookupOrCreateBertAttentionNVGPUF32Fn(moduleOp);
    } else if (elementType.isF16()) {
      attnFn = lookupOrCreateBertAttentionNVGPUF16Fn(moduleOp);
    } else {
      llvm_unreachable("Not ok");
    }

    func::FuncOp allocFn;

    if (returnType.getElementType().isF32()) {
      allocFn = lookupOrCreateAlloc3DMemRefNVGPUF32Fn(moduleOp);
    } else if (returnType.getElementType().isF16()) {
      allocFn = lookupOrCreateAlloc3DMemRefNVGPUF16Fn(moduleOp);
    } else {
      llvm_unreachable("Not good.");
    }

    // To-do: Stupid Static Shape Inference Here. Should convert to dynamic
    // shape if I have time.
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

    auto scale = rewriter.create<arith::ConstantFloatOp>(loc, op.getScale(),
                                                         rewriter.getF32Type());
    auto headNum = rewriter.create<arith::ConstantIntOp>(loc, op.getHeadNum(),
                                                         rewriter.getI64Type());

    SmallVector<Value> operands = {op.getOperand(0), op.getOperand(1),
                                   allocCaller.getResult(0),
                                   scale->getResult(0), headNum.getResult()};
    auto attnCaller = rewriter.create<func::CallOp>(loc, attnFn, operands);

    while (!op.use_empty()) {
      op->getUses().begin()->set(attnCaller.getOperand(2));
    }

    rewriter.eraseOp(op);

    return success();
  }
};

LogicalResult HOMNVGPUToFuncPass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);
  patterns = std::move(patternList);
  return success();
}

void HOMNVGPUToFuncPass::runOnOperation() {
  auto *context = &getContext();
  RewritePatternSet convPatterns(context);
  ConversionTarget target(*context);

  HOMFuncTypeConverter typeConverter;

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

  convPatterns.add<ConvertHOMNVGPUMatmulOp, ConvertHOMConstantOp,
                   ConvertHOMNVGPUBertMhaOp, ConvertHOMNVGPULayernormOp,
                   ConvertHOMNVGPUAddOp, ConvertHOMNVGPUGatherOp,
                   ConvertHOMDummyTensorOp, ConvertHOMNVGPUCuSeqLenOp>(
      typeConverter, context);

  target.addLegalDialect<func::FuncDialect, arith::ArithDialect>();
  target.addIllegalDialect<hom::HOMDialect, homnvgpu::HOMNVGPUDialect>();

  target.addLegalOp<hom::DummyTensorOp>();

  if (failed(applyFullConversion(getOperation(), target,
                                 std::move(convPatterns)))) {
    signalPassFailure();
  }
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

} // namespace
} // namespace hands_on_mlir
} // namespace mlir
