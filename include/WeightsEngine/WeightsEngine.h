#ifndef HANDS_ON_MLIR_WEIGHTSENGINE_WEIGHTSENGINE_H_
#define HANDS_ON_MLIR_WEIGHTSENGINE_WEIGHTSENGINE_H_

#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include <cstddef>
#include <memory>
#include <unordered_map>

namespace mlir {
namespace hands_on_mlir {
class WeightsEngine {
  size_t weightsIds;
  std::unordered_map<size_t, std::shared_ptr<void>> weightsMap;

public:
  WeightsEngine() { weightsIds = 0; }
  size_t addWeight(std::shared_ptr<void>);
  size_t addWeight(ElementsAttr &);
  void removeWeight(size_t idx);

  template <class T>
  static void serializeWeightToDisk(const ShapedType &shape, void *data,
                                    const std::string &fileName);
};

extern WeightsEngine gWe;

} // namespace hands_on_mlir
} // namespace mlir

#endif
