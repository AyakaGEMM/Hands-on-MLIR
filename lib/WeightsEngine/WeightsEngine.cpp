#include "WeightsEngine/WeightsEngine.h"
#include "half.hpp"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

using half_float::half;

namespace mlir {
namespace hands_on_mlir {

template <class T> void printElement(const T &element, llvm::raw_ostream &out) {
  element.print(out);
}

template <>
void printElement<APInt>(const APInt &element, llvm::raw_ostream &out) {
  element.print(out, true);
}

template <class T, class T0> T convertToT0(T0 data) {
  llvm_unreachable("Unsupported Type");
}

template <> float convertToT0(APFloat data) { return data.convertToFloat(); }
template <> half convertToT0(APFloat data) {
  return half(data.convertToFloat());
}

template <class T> T convertToT0(APInt data) { return data.getLimitedValue(); }

template <class T, class T0>
void WeightsEngine::castElementsToPtr(ElementsAttr &element, T0 *ptr) {
  auto data = element.getValues<T>();

  for (int i = 0; i < element.getNumElements(); i++) {
    ptr[i] = convertToT0<T0>(data[i]);
  }
}

template <class T>
void WeightsEngine::serializeWeightToDisk(ElementsAttr &value,
                                          const std::string &fileName) {
  auto shape = value.getShapedType();
  auto data = value.getValues<T>();
  auto dimSize = shape.getShape();
  std::error_code EC;
  llvm::raw_fd_ostream out(fileName, EC);
  for (auto i : dimSize) {
    out << i << " ";
  }
  out << "\n";
  auto totalSize = value.getNumElements();
  for (int i = 0; i < totalSize; i++) {
    printElement(data[i], out);
  }
  out << "\n";
}

size_t WeightsEngine::addWeight(std::shared_ptr<void> weight) {
  weightsMap[weightsIds++] = weight;
  return weightsIds - 1;
}

size_t WeightsEngine::addWeight(ElementsAttr &element) {
  auto shapeType = element.getShapedType();
  auto elementType = element.getElementType();

  void *dataPtr = malloc(shapeType.getElementTypeBitWidth() / 8 *
                         shapeType.getNumElements());
  std::shared_ptr<void> sPtr;
  if (elementType.isF32()) {
    castElementsToPtr<APFloat>(element, static_cast<float *>(dataPtr));
    sPtr.reset(static_cast<float *>(dataPtr), free);
  } else if (elementType.isF16()) {
    castElementsToPtr<APFloat>(element, static_cast<half *>(dataPtr));
    sPtr.reset(static_cast<float *>(dataPtr), free);
  } else if (elementType.isIntOrIndex()) {
    auto intType = llvm::dyn_cast<IntegerType>(elementType);
    switch (intType.getWidth()) {
    case 64:
      castElementsToPtr<APInt>(element, static_cast<size_t *>(dataPtr));
      sPtr.reset(static_cast<size_t *>(dataPtr), free);
      break;
    case 32:
      castElementsToPtr<APInt>(element, static_cast<int32_t *>(dataPtr));
      sPtr.reset(static_cast<int32_t *>(dataPtr), free);
      break;
    case 8:
      castElementsToPtr<APInt>(element, static_cast<int8_t *>(dataPtr));
      sPtr.reset(static_cast<int8_t *>(dataPtr), free);
      break;
    default:
      llvm_unreachable("Unsupported integer width. ");
    }
  } else {
    llvm_unreachable("Unsupported Type");
  }
  auto idx = addWeight(sPtr);
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
