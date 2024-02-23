#include "WeightsEngine/WeightsEngine.h"
#include "WeightsEngine/Utils.h"
#include "half.h"
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
namespace mlir {
namespace hands_on_mlir {

template <class T>
static void printNativeElement(const T &element, llvm::raw_ostream &out) {
  out << element;
}

template <>
void printNativeElement<fp16>(const fp16 &element, llvm::raw_ostream &out) {
  out << float(element);
}

template <class T>
void WeightsEngine::serializeWeightToDisk(const ShapedType &shape, T *data,
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
  for (int i = 0; i < totalSize; i++) {
    printNativeElement(data[i], out);
    out << " ";
  }
  out << "\n";
}

size_t WeightsEngine::addWeight(std::shared_ptr<void> weight) {
  weightsMap[weightsIds++] = weight;
  return weightsIds - 1;
}

size_t WeightsEngine::addWeight(ElementsAttr &elements) {
  std::shared_ptr<void> sPtr;
  auto idx = addWeight(sPtr);

  auto fn = [&]<typename T>(std::shared_ptr<T> dataPtr) {
    serializeWeightToDisk<T>(
        elements.getShapedType(), dataPtr.get(),
        std::filesystem::path(__FILE__).parent_path().string() +
            std::string("/../../examples/torch/linear/") + std::to_string(idx) +
            ".txt");
    sPtr = dataPtr;
  };

  universalCastElementsToPtr(elements, fn);

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
