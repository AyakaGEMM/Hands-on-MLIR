#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace vector;

namespace {

struct ValueToRange { // Work around for Value to Range convertion.
  SmallVector<Value> v_vector;
  ArrayRef<Value> ref;
  ValueRange vr;
  ValueToRange(Value &v) : v_vector(1, v), ref(v_vector), vr(ref) {}
  ValueToRange(const Value &v) : v_vector(1, v), ref(v_vector), vr(ref) {}
};

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

    // Create Constant
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    // Create M,N,K
    Value M = rewriter.create<memref::DimOp>(loc, A, c0);
    Value N = rewriter.create<memref::DimOp>(loc, C, c1);
    Value K = rewriter.create<memref::DimOp>(loc, B, c0);

    ValueToRange M_range(M), c0_range(c0);

    AffineForOp M_loop, N_loop, K_loop;

    M_loop = rewriter.create<AffineForOp>(
        loc, c0_range.vr, rewriter.getDimIdentityMap(), M_range.vr,
        rewriter.getDimIdentityMap(), 1, std::nullopt,
        [&](OpBuilder &builder, Location loc, Value im, ValueRange iterArgs) {
          ValueToRange N_range(N);
          N_loop = builder.create<AffineForOp>(
              loc, c0_range.vr, builder.getDimIdentityMap(), N_range.vr,
              builder.getDimIdentityMap(), 1, std::nullopt,
              [&](OpBuilder &builder, Location loc, Value in,
                  ValueRange iterArgs) {
                ValueToRange K_range(K);
                K_loop = builder.create<AffineForOp>(
                    loc, c0_range.vr, builder.getDimIdentityMap(), K_range.vr,
                    builder.getDimIdentityMap(), 1, std::nullopt,
                    [&](OpBuilder &builder, Location loc, Value ik,
                        ValueRange iterArgs) {
                      SmallVector<Value> load_A_mem_indices, load_B_mem_indices,
                          load_C_mem_indices;
                      load_A_mem_indices.push_back(im);
                      load_A_mem_indices.push_back(ik);
                      load_B_mem_indices.push_back(ik);
                      load_B_mem_indices.push_back(in);
                      load_C_mem_indices.push_back(im);
                      load_C_mem_indices.push_back(in);
                      Value a = builder.create<AffineLoadOp>(
                          loc, A, load_A_mem_indices);
                      Value b = builder.create<AffineLoadOp>(
                          loc, B, load_B_mem_indices);
                      Value c = builder.create<AffineLoadOp>(
                          loc, C, load_C_mem_indices);
                      Value resc = builder.create<math::FmaOp>(loc, a, b, c);
                      builder.create<AffineStoreOp>(loc, resc, C,
                                                    load_C_mem_indices);
                      builder.create<AffineYieldOp>(loc);
                    });
                Attribute K_Attr = rewriter.getStringAttr("K_loop");
                K_loop->setAttr("Dimension", K_Attr);
                builder.create<AffineYieldOp>(loc);
              });
          Attribute N_Attr = rewriter.getStringAttr("N_loop");
          N_loop->setAttr("Dimension", N_Attr);
          builder.create<AffineYieldOp>(loc);
        });

    Attribute M_Attr = rewriter.getStringAttr("M_loop");
    M_loop->setAttr("Dimension", M_Attr);

    interchangeLoops(N_loop, K_loop); // naive optimization
    interchangeLoops(M_loop, K_loop);

    rewriter.eraseOp(op);
    return success();
  }

  const size_t M_KERNEL_SIZE = 6;
  const size_t N_KERNEL_SIZE = 16;
  const int32_t K_BLOCK_SIZE = 1024;
  const int32_t M_BLOCK_SIZE = 384;
  const int32_t N_BLOCK_SIZE = 1024;
}; // namespace
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
    registry.insert<AffineDialect, func::FuncDialect, memref::MemRefDialect,
                    math::MathDialect>();
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
                         func::FuncDialect, memref::MemRefDialect,
                         math::MathDialect>();
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