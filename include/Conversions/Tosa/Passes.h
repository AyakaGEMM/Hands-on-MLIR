#ifndef HOM_CONVERSIONS_TOSA_TRANSFORMS_PASSES_H
#define HOM_CONVERSIONS_TOSA_TRANSFORMS_PASSES_H

#include <memory>
#include <numeric>

#include "WeightsEngine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DECL_TOSATOHOMPASS
#define GEN_PASS_DECL_TOSACONSTANTFOLDINGPASS
#define GEN_PASS_REGISTRATION
#include "Conversions/Tosa/Passes.h.inc"

template <typename T> std::shared_ptr<T[]> getNewData(size_t size) {
  static std::shared_ptr<T[]> newData;
  static size_t newDataSize = 0;
  if (size > newDataSize) {
    newData.reset(new T[size]);
    newDataSize = size;
  }
  return newData;
}

template <typename T, typename T0>
auto doCastFolding(T *data, Type newType, ArrayRef<int64_t> size,
                   size_t totalSize) {
  auto newData = getNewData<T0>(totalSize);
  for (size_t i = 0; i < totalSize; i++) {
    newData.get()[i] = data[i];
  }
  return getDenseElementsAttr(newType, size, newData.get(), totalSize);
}

template <typename T>
tosa::ConstOp foldCast(PatternRewriter &rewriter, tosa::ConstOp oldConst,
                       Type newType, T *data, ArrayRef<int64_t> size) {
  auto requestNewDataSize =
      std::accumulate(size.begin(), size.end(), 1, std::multiplies<T>());

  DenseElementsAttr denseAttr;

  if (newType.isF32()) {
    denseAttr =
        doCastFolding<T, float>(data, newType, size, requestNewDataSize);
  } else if (newType.isF16()) {
    denseAttr = doCastFolding<T, fp16>(data, newType, size, requestNewDataSize);
  } else if (newType.isIntOrIndex()) {
    auto intType = llvm::dyn_cast<IntegerType>(newType);
    switch (intType.getWidth()) {
    case 64:
      denseAttr =
          doCastFolding<T, int64_t>(data, newType, size, requestNewDataSize);
      break;
    case 32:
      denseAttr =
          doCastFolding<T, int32_t>(data, newType, size, requestNewDataSize);
      break;
    case 16:
      denseAttr =
          doCastFolding<T, int16_t>(data, newType, size, requestNewDataSize);
      break;
    case 8:
      denseAttr =
          doCastFolding<T, int8_t>(data, newType, size, requestNewDataSize);
      break;
    default:
      llvm_unreachable("Unsupported integer width. ");
    }
  } else {
    llvm_unreachable("Not supported type.");
  }

  return rewriter.create<tosa::ConstOp>(oldConst->getLoc(), denseAttr.getType(),
                                        denseAttr);
}

inline void registerTosaToHOMPipelines() {
  PassPipelineRegistration<>(
      "tosa-to-hom-pipeline",
      "Convert TOSA operators to hom with some optimization",
      [](OpPassManager &pm) {
        tosa::TosaLayerwiseConstantFoldPassOptions tosaConstFoldOption;
        tosaConstFoldOption.aggressiveReduceConstant = true;
        pm.addPass(
            tosa::createTosaLayerwiseConstantFoldPass(tosaConstFoldOption));
        pm.addPass(createTosaConstantFoldingPass());
        pm.addPass(
            tosa::createTosaLayerwiseConstantFoldPass(tosaConstFoldOption));
        pm.addPass(createTosaToHOMPass());
      });
}

} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir

#endif // HOM_CONVERSIONS_TOSA_TRANSFORMS_PASSES_H
