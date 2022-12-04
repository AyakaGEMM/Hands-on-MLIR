#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
struct MatMulCPUOptimize : public ConversionPattern {
  MatMulCPUOptimize(MLIRContext *ctx)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Get input A, B, C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);
    // Get shape of input and output
    ShapedType ATy = A.getType().cast<ShapedType>();

    // Some constants.
    const Value i = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    const AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    const AffineExpr s1 = rewriter.getAffineSymbolExpr(1);
    const AffineExpr d0 = rewriter.getAffineDimExpr(0);
    const AffineExpr d1 = rewriter.getAffineDimExpr(1);
    const AffineMap mapBroadcast =
        AffineMap::get(2, 0, rewriter.getAffineConstantExpr(0));
    const VectorType vTy = VectorType::get(16, ATy.getElementType());

    // Dims
    Value M = rewriter.create<memref::DimOp>(loc, A, 0);
    Value N = rewriter.create<memref::DimOp>(loc, B, 1);
    Value K = rewriter.create<memref::DimOp>(loc, A, 1);

    buildAffineLoopNest( // Loop K
        rewriter, loc, {i}, {K}, K_BLOCK_SIZE,
        [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
          Value ik = ivRange.back();
          const Value i = rewriter.create<arith::ConstantIndexOp>(loc, 0);
          buildAffineLoopNest( // Loop N
              builder, loc, {i}, {N}, N_BLOCK_SIZE,
              [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                Value in = ivRange.back();
                const Value i = rewriter.create<arith::ConstantIndexOp>(loc, 0);
                buildAffineLoopNest(
                    builder, loc, {i}, {M}, // Loop M
                    M_BLOCK_SIZE,
                    [&](OpBuilder &builder, Location loc, ValueRange ivRange) {
                      Value im = ivRange.back();
                    });
              });
        });

    rewriter.eraseOp(op);
    return success();
  }

  const size_t M_KERNEL_SIZE = 6;
  const size_t N_KERNEL_SIZE = 16;
  const size_t K_BLOCK_SIZE = 1024;
  const size_t M_BLOCK_SIZE = 384;
  const size_t N_BLOCK_SIZE = 1024;
};
} // namespace

namespace {
struct MatMulCPUOptimizePass
    : public PassWrapper<MatMulCPUOptimizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulCPUOptimizePass)

  StringRef getArgument() const final { return "matmul-cpu-optimize"; }
  StringRef getDescription() const final {
    return "MatMul Optimization on CPU.";
  }

  MatMulCPUOptimizePass() = default;
  MatMulCPUOptimizePass(const MatMulCPUOptimizePass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, func::FuncDialect, memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void MatMulCPUOptimizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect>();
  target.addIllegalOp<linalg::MatmulOp>();

  RewritePatternSet patterns(context);
  patterns.add<MatMulCPUOptimize>(context);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace hands_on_mlir {
void registerMatMulCPUOptimizePass() {
  PassRegistration<MatMulCPUOptimizePass>();
}
} // namespace hands_on_mlir
} // namespace mlir