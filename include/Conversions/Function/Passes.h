#ifndef HOM_CONVERSIONS_FUNC_TRANSFORMS_PASSES_H
#define HOM_CONVERSIONS_FUNC_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Conversions/Function/FunctionUtils.h"
#include "HOM/HOMOps.h"

namespace mlir {
namespace hands_on_mlir {

#define GEN_PASS_DECL_EXTRACTINITFUNCPASS
#define GEN_PASS_DECL_HOMTOFUNCPASS
#define GEN_PASS_DECL_HOMNVGPUTOFUNCPASS
#define GEN_PASS_DECL_UNIFYLLVMFUNCINTERFACEPASS
#define GEN_PASS_DECL_OPTIMIZEMEMORYPASS
#define GEN_PASS_REGISTRATION
#include "Conversions/Function/Passes.h.inc"

namespace {
struct ConvertHOMDummyTensorOp
    : public OpConversionPattern<hom::DummyTensorOp> {
  using OpConversionPattern<hom::DummyTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hom::DummyTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op->getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto allocFn = lookupOrCreateAllocDummyTensorF32Fn(moduleOp);

    auto allocCaller =
        rewriter.create<func::CallOp>(loc, allocFn, ArrayRef<Value>{});

    while (!op.use_empty()) {
      op->getUses().begin()->set(allocCaller->getResult(0));
    }

    // maybeInsertDeallocFn(rewriter, op, {allocCaller->getResult(0)});
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

} // namespace hands_on_mlir
} // namespace mlir

#endif // HOM_CONVERSIONS_FUNC_TRANSFORMS_PASSES_H
