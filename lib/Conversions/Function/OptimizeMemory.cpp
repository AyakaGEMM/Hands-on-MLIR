#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "Conversions/Function/FunctionUtils.h"
#include "Conversions/Function/Passes.h"
#include "HOM/HOMOps.h"
#include "WeightsEngine/WeightsEngine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define PASS_NAME "unify-llvm-func-interface"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {

#define GEN_PASS_DEF_OPTIMIZEMEMORYPASS
#include "Conversions/Function/Passes.h.inc"

namespace {

struct OptimizeMemoryPass : impl::OptimizeMemoryPassBase<OptimizeMemoryPass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

struct OptimizeMemory : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

LogicalResult OptimizeMemoryPass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);
  patterns = std::move(patternList);
  return success();
}

void OptimizeMemoryPass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

} // namespace
} // namespace hands_on_mlir
} // namespace mlir
