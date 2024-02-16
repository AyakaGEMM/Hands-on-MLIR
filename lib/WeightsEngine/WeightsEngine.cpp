#include "WeightsEngine/WeightsEngine.h"
#include "WeightsEngine/Utils.h"
#include "half.hpp"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

using half_float::half;

namespace mlir {
namespace hands_on_mlir {

template <class T>
void WeightsEngine::serializeWeightToDisk(const ShapedType &shape,
                                          void *dataPtr,
                                          const std::string &fileName) {
  auto dimSize = shape.getShape();
  std::error_code EC;
  llvm::raw_fd_ostream out(fileName, EC);
  // To-do: Change to a better store format.
  for (auto i : dimSize) {
    out << i << " ";
  }
  out << "\n";
  auto totalSize = shape.getNumElements();
  auto data = static_cast<T *>(dataPtr);
  for (int i = 0; i < totalSize; i++) {
    out << data[i] << " ";
  }
  out << "\n";
}

size_t WeightsEngine::addWeight(std::shared_ptr<void> weight) {
  weightsMap[weightsIds++] = weight;
  return weightsIds - 1;
}

size_t WeightsEngine::addWeight(ElementsAttr &element) {
  auto elementType = element.getElementType();

  void *dataPtr;
  std::shared_ptr<void> sPtr;
  auto idx = addWeight(sPtr);
  if (elementType.isF32()) {
    castElementsToPtr<APFloat, float>(element, &dataPtr);
    // To-do: Make it configurable
    serializeWeightToDisk<float>(
        element.getShapedType(), dataPtr,
        std::filesystem::path(__FILE__).parent_path().string() +
            std::string("/../../examples/torch/linear/") + std::to_string(idx) +
            ".txt");
    sPtr.reset(static_cast<float *>(dataPtr), free);
  } else if (elementType.isF16()) {
    castElementsToPtr<APFloat, half_float::half>(element, &dataPtr);
    sPtr.reset(static_cast<half_float::half *>(dataPtr), free);
  } else if (elementType.isIntOrIndex()) {
    auto intType = llvm::dyn_cast<IntegerType>(elementType);
    switch (intType.getWidth()) {
    case 64:
      castElementsToPtr<APInt, int64_t>(element, &dataPtr);
      sPtr.reset(static_cast<int64_t *>(dataPtr), free);
      break;
    case 32:
      castElementsToPtr<APInt, int32_t>(element, &dataPtr);
      sPtr.reset(static_cast<int32_t *>(dataPtr), free);
      break;
    case 16:
      castElementsToPtr<APInt, int16_t>(element, &dataPtr);
      sPtr.reset(static_cast<int16_t *>(dataPtr), free);
      break;
    case 8:
      castElementsToPtr<APInt, int8_t>(element, &dataPtr);
      sPtr.reset(static_cast<int8_t *>(dataPtr), free);
      break;
    default:
      llvm_unreachable("Unsupported integer width. ");
    }
  } else {
    llvm_unreachable("Unsupported Type");
  }
  return idx;
}

void WeightsEngine::removeWeight(size_t idx) {
  auto iter = weightsMap.find(idx);
  if (iter != weightsMap.end()) {
    weightsMap.erase(iter);
  }
}

WeightsEngine gWe;

} // namespace hands_on_mlir
} // namespace mlir
