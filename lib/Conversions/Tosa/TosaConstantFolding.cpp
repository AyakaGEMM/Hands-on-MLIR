#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "Conversions/Tosa/Passes.h"
#include "HOM/HOMOps.h"
#include "WeightsEngine/Utils.h"
#include "WeightsEngine/WeightsEngine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#define PASS_NAME "tosa-to-hom"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DEF_TOSACONSTANTFOLDINGPASS
#include "Conversions/Tosa/Passes.h.inc"

namespace {

template <typename T>
void doSliceFolding(const T *oldData, T *newData, ArrayRef<int64_t> start,
                    ArrayRef<int64_t> size, ArrayRef<int64_t> dataSize,
                    size_t dim) {
  if (dim == dataSize.size()) {
    newData[0] = oldData[0];
    return;
  }
  auto oldStride = std::accumulate(dataSize.begin() + dim + 1, dataSize.end(),
                                   1, std::multiplies<int64_t>());
  auto newStride = std::accumulate(size.begin() + dim + 1, size.end(), 1,
                                   std::multiplies<int64_t>());
  for (int64_t i = 0; i < size[dim]; i++) {
    doSliceFolding(oldData + (start[dim] + i) * oldStride,
                   newData + i * newStride, start, size, dataSize, dim + 1);
  }
}

template <typename T>
static tosa::ConstOp foldSlice(PatternRewriter &rewriter,
                               tosa::ConstOp oldConst, T *data,
                               ArrayRef<int64_t> start, ArrayRef<int64_t> size,
                               ArrayRef<int64_t> dataSize) {
  auto oldOutput = oldConst.getOutput();
  auto requestNewDataSize =
      std::accumulate(size.begin(), size.end(), 1, std::multiplies<T>());
  auto newData = getNewData<T>(requestNewDataSize);

  doSliceFolding(data, newData.get(), start, size, dataSize, 0);

  auto denseAttr =
      getDenseElementsAttr(oldOutput.getType().getElementType(), size,
                           newData.get(), requestNewDataSize);

  return rewriter.create<tosa::ConstOp>(oldConst->getLoc(), denseAttr.getType(),
                                        denseAttr);
}

template <typename T>
static tosa::ConstOp
foldGather(PatternRewriter &rewriter, tosa::ConstOp oldConst, T *data,
           int32_t *indices, int64_t indicesStride, ArrayRef<int64_t> size,
           ArrayRef<int64_t> stride, ArrayRef<int64_t> dataSize,
           ArrayRef<int64_t> dataStride) {
  auto oldOutput = oldConst.getOutput();
  auto requestNewDataSize =
      std::accumulate(size.begin(), size.end(), 1, std::multiplies<T>());
  auto newData = getNewData<T>(requestNewDataSize);

  for (int64_t i = 0; i < size[0]; i++) {
    for (int64_t j = 0; j < size[1]; j++) {
      for (int64_t k = 0; k < size[2]; k++) {
        newData[i * stride[0] + j * stride[1] + k] =
            data[i * dataStride[0] +
                 indices[i * indicesStride + j] * dataStride[1] + k];
      }
    }
  }

  auto denseAttr =
      getDenseElementsAttr(oldOutput.getType().getElementType(), size,
                           newData.get(), requestNewDataSize);

  return rewriter.create<tosa::ConstOp>(oldConst->getLoc(), denseAttr.getType(),
                                        denseAttr);
}

struct FoldTosaConstOp : public OpRewritePattern<tosa::ConstOp> {
  using OpRewritePattern<tosa::ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ConstOp op,
                                PatternRewriter &rewriter) const override {
    auto elements = op.getValueAttr();

    if (op->hasOneUse()) {
      auto user = *(op->getUsers().begin());
      if (auto sliceOp = dyn_cast<tosa::SliceOp>(user)) {
        auto start = sliceOp.getStart();
        auto size = sliceOp.getSize();
        auto dataSize = sliceOp.getInput().getType().getShape();

        tosa::ConstOp newOp;

        auto fn = [&]<typename T>(std::shared_ptr<T> dataPtr) {
          newOp =
              foldSlice<T>(rewriter, op, dataPtr.get(), start, size, dataSize);
        };

        universalCastElementsToPtr(elements, fn);
        sliceOp.replaceAllUsesWith(newOp.getOutput());
        return success();
      } else if (auto castOp = dyn_cast<tosa::CastOp>(user)) {
        tosa::ConstOp newOp;
        auto fn = [&]<typename T>(std::shared_ptr<T> dataPtr) {
          newOp = foldCast<T>(
              rewriter, op, castOp.getOutput().getType().getElementType(),
              dataPtr.get(), castOp.getOutput().getType().getShape());
        };

        universalCastElementsToPtr(elements, fn);
        castOp.replaceAllUsesWith(newOp.getOutput());
        return success();
      } else if (auto gatherOp = dyn_cast<tosa::GatherOp>(user)) {
        auto indices =
            dyn_cast<tosa::ConstOp>(gatherOp.getIndices().getDefiningOp());
        auto values =
            dyn_cast<tosa::ConstOp>(gatherOp.getValues().getDefiningOp());

        if (indices && values &&
            dyn_cast<RankedTensorType>(gatherOp.getOutput().getType())) {
          tosa::ConstOp newOp;
          void *indicesPtr = nullptr;
          castElementsToPtr<APInt, int32_t>(indices.getValue(), &indicesPtr);
          auto indicesShape = indices.getValue().getShapedType().getShape();
          auto fn = [&]<typename T>(std::shared_ptr<T> dataPtr) {
            auto indicesValue = static_cast<int32_t *>(indicesPtr);
            auto valuesShape = values.getValue().getShapedType().getShape();
            auto outputShape =
                dyn_cast<RankedTensorType>(gatherOp.getOutput().getType())
                    .getShape();
            newOp =
                foldGather<T>(rewriter, values, dataPtr.get(), indicesValue,
                              indicesShape[1], outputShape,
                              ArrayRef<int64_t>{outputShape[2] * outputShape[1],
                                                outputShape[2], 1},
                              valuesShape,
                              ArrayRef<int64_t>{valuesShape[2] * valuesShape[1],
                                                valuesShape[2], 1});
            free(indicesPtr);
          };

          universalCastElementsToPtr(values.getValue(), fn);
          gatherOp.replaceAllUsesWith(newOp.getOutput());
          return success();
        }
      }
    }

    return failure();
  }
};

struct TosaConstantFoldingPass
    : impl::TosaConstantFoldingPassBase<TosaConstantFoldingPass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

LogicalResult TosaConstantFoldingPass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);

  patternList.add<FoldTosaConstOp>(ctx);
  patterns = std::move(patternList);
  return success();
}

void TosaConstantFoldingPass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

} // namespace

} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir
