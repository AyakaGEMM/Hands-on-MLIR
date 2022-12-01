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
class MatMulCPUOptimize : public ConversionPattern {
  MatMulCPUOptimize(MLIRContext *ctx)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    return success();
  }
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

  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect>();
}

namespace mlir {
namespace hands_on_mlir {
void registerMatMulCPUOptimizePass() {
  PassRegistration<MatMulCPUOptimizePass>();
}
} // namespace hands_on_mlir
} // namespace mlir