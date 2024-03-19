#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "Conversions/Tosa/Passes.h"
#include "HOM/HOMOps.h"
#include "HOMNVGPU/HOMNVGPUOps.h"
#include "WeightsEngine/WeightsEngine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define PASS_NAME "homnvgpu-legalize-gemm"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu {

#define GEN_PASS_DEF_HOMNVGPULEGALIZEGEMMPASS
#include "HOMNVGPU/HOMNVGPULegalizeGemm.pdll.h.inc"
#include "HOMNVGPU/Passes.h.inc"

namespace {

static void generateTransposeImpl(PatternRewriter &rewriter,
                                  Operation *matmul_) {
  auto matmul = dyn_cast<homnvgpu::MatmulOp>(matmul_);

  if (auto defining = matmul->getOperand(1).getDefiningOp()) {
    if (auto constOp = dyn_cast<tosa::ConstOp>(defining)) {
      auto oldType = constOp.getResult().getType();
      auto oldShape = oldType.getShape();

      assert(oldShape.size() == 3);

      SmallVector<int64_t> newShape = {oldShape[0], oldShape[2], oldShape[1]};

      tosa::TransposeOp transposeOp;
      auto needCreateTranspose = [&transposeOp, &constOp]() {
        for (const auto &user : constOp->getUsers()) {
          if (auto op = dyn_cast<tosa::TransposeOp>(user)) {
            llvm::SmallVector<int64_t> perms;
            if (op.getConstantPerms(perms).succeeded() && perms[0] == 0 &&
                perms[1] == 2 && perms[2] == 1) {
              transposeOp = op;
              return false;
            }
          }
        }
        return true;
      };

      if (needCreateTranspose()) {
        auto permAttr = DenseIntElementsAttr::get(
            RankedTensorType::get({3}, rewriter.getI32Type()),
            ArrayRef<int32_t>{0, 2, 1});

        rewriter.setInsertionPointToStart(matmul->getBlock());
        auto perm = rewriter.create<tosa::ConstOp>(
            constOp->getLoc(), permAttr.getType(), permAttr);
        transposeOp = rewriter.create<tosa::TransposeOp>(
            constOp->getLoc(),
            RankedTensorType::get(newShape, oldType.getElementType()),
            constOp.getResult(), perm.getResult());
      }

      matmul->setOperand(1, transposeOp.getResult());
      matmul.setTransb(true);
      return;
    } else if (auto transposeOp = dyn_cast<tosa::TransposeOp>(
                   matmul->getOperand(1).getDefiningOp())) {
      auto perm =
          dyn_cast<tosa::ConstOp>(transposeOp.getPerms().getDefiningOp())
              .getValue()
              .getValues<int32_t>();
      if (perm.size() == 3 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1) {
        matmul->setOperand(1, transposeOp.getInput1());
        matmul.setTransb(true);
        return;
      }
    }
  }
  llvm_unreachable("Does not support this format.");
}

struct HOMNVGPULegalizeGemmPass
    : impl::HOMNVGPULegalizeGemmPassBase<HOMNVGPULegalizeGemmPass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

LogicalResult HOMNVGPULegalizeGemmPass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);

  populateGeneratedPDLLPatterns(patternList);
  patternList.getPDLPatterns().registerRewriteFunction("generateTranspose",
                                                       generateTransposeImpl);
  patterns = std::move(patternList);
  return success();
}

void HOMNVGPULegalizeGemmPass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

} // namespace
} // namespace homnvgpu
} // namespace hands_on_mlir
} // namespace mlir
