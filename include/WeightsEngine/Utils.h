#ifndef HANDS_ON_MLIR_WEIGHTSENGINE_UTILS_H_
#define HANDS_ON_MLIR_WEIGHTSENGINE_UTILS_H_

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/tfloat32.h"
#include "half.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/VersionTuple.h"
#include <cassert>
#include <cstdint>
#include <memory>

namespace mlir {
namespace hands_on_mlir {

template <class T> void printElement(const T &element, llvm::raw_ostream &out) {
  element.print(out);
}

template <>
inline void printElement<APInt>(const APInt &element, llvm::raw_ostream &out) {
  element.print(out, true);
}

template <class T, class T0> T convertToT0(T0 data) {
  llvm_unreachable("Unsupported Type");
}

template <> inline double convertToT0(APFloat data) {
  return data.convertToDouble();
}

template <> inline float convertToT0(APFloat data) {
  return data.convertToFloat();
}

template <> inline fp16 convertToT0(APFloat data) {
  return fp16(data.convertToFloat());
}

template <class T> T convertToT0(APInt data) { return data.getLimitedValue(); }

template <class T, class T0>
void castElementsToPtr(const ElementsAttr &element, void **ptrptr) {
  auto data = element.getValues<T>();
  auto shapeType = element.getShapedType();

  auto &ptr = *ptrptr;
  ptr = malloc(shapeType.getElementTypeBitWidth() / 8 *
               shapeType.getNumElements());
  auto ptrWithType = static_cast<T0 *>(ptr);

  for (int i = 0; i < element.getNumElements(); i++) {
    ptrWithType[i] = convertToT0<T0>(data[i]);
  }
}

void universalCastElementsToPtr(const ElementsAttr &elements, auto &&fn = [
]<typename T = void>(std::shared_ptr<T> dataPtr){}) {
  void *dataPtr = nullptr;
  auto elementType = elements.getElementType();
  if (elementType.isF32()) {
    castElementsToPtr<APFloat, float>(elements, &dataPtr);
    fn.template operator()<float>(
        std::shared_ptr<float>(static_cast<float *>(dataPtr), free));
  } else if (elementType.isF16()) {
    castElementsToPtr<APFloat, fp16>(elements, &dataPtr);
    fn.template operator()<fp16>(
        std::shared_ptr<fp16>(static_cast<fp16 *>(dataPtr), free));
  } else if (elementType.isIntOrIndex()) {
    auto intType = llvm::dyn_cast<IntegerType>(elementType);
    switch (intType.getWidth()) {
    case 64:
      castElementsToPtr<APInt, int64_t>(elements, &dataPtr);
      fn.template operator()<int64_t>(
          std::shared_ptr<int64_t>(static_cast<int64_t *>(dataPtr), free));
      break;
    case 32:
      castElementsToPtr<APInt, int32_t>(elements, &dataPtr);
      fn.template operator()<int32_t>(
          std::shared_ptr<int32_t>(static_cast<int32_t *>(dataPtr), free));
      break;
    case 16:
      castElementsToPtr<APInt, int16_t>(elements, &dataPtr);
      fn.template operator()<int16_t>(
          std::shared_ptr<int16_t>(static_cast<int16_t *>(dataPtr), free));
      break;
    case 8:
      castElementsToPtr<APInt, int8_t>(elements, &dataPtr);
      fn.template operator()<int8_t>(
          std::shared_ptr<int8_t>(static_cast<int8_t *>(dataPtr), free));
      break;
    default:
      llvm_unreachable("Unsupported integer width. ");
    }
  }
}

void universalCastElementsToPtr(
    const ElementsAttr &elementsA, const ElementsAttr &elementsB,
    auto &&fn = []<typename T = void>(std::shared_ptr<T> dataPtrA,
                                      std::shared_ptr<T> dataPtrB){}) {
  void *dataPtrA = nullptr;
  void *dataPtrB = nullptr;
  auto elementType = elementsA.getElementType();
  auto elementTypeB = elementsB.getElementType();
  assert(elementType == elementTypeB);
  if (elementType.isF32()) {
    castElementsToPtr<APFloat, float>(elementsA, &dataPtrA);
    castElementsToPtr<APFloat, float>(elementsB, &dataPtrB);
    fn.template operator()<float>(
        std::shared_ptr<float>(static_cast<float *>(dataPtrA), free),
        std::shared_ptr<float>(static_cast<float *>(dataPtrB), free));
  } else if (elementType.isF16()) {
    castElementsToPtr<APFloat, fp16>(elementsA, &dataPtrA);
    castElementsToPtr<APFloat, fp16>(elementsB, &dataPtrB);
    fn.template operator()<fp16>(
        std::shared_ptr<fp16>(static_cast<fp16 *>(dataPtrA), free),
        std::shared_ptr<fp16>(static_cast<fp16 *>(dataPtrB), free));
  } else if (elementType.isIntOrIndex()) {
    auto intType = llvm::dyn_cast<IntegerType>(elementType);
    switch (intType.getWidth()) {
    case 64:
      castElementsToPtr<APInt, int64_t>(elementsA, &dataPtrA);
      castElementsToPtr<APInt, int64_t>(elementsB, &dataPtrB);
      fn.template operator()<int64_t>(
          std::shared_ptr<int64_t>(static_cast<int64_t *>(dataPtrA), free),
          std::shared_ptr<int64_t>(static_cast<int64_t *>(dataPtrB), free));
      break;
    case 32:
      castElementsToPtr<APInt, int32_t>(elementsA, &dataPtrA);
      castElementsToPtr<APInt, int32_t>(elementsB, &dataPtrB);
      fn.template operator()<int32_t>(
          std::shared_ptr<int32_t>(static_cast<int32_t *>(dataPtrA), free),
          std::shared_ptr<int32_t>(static_cast<int32_t *>(dataPtrB), free));
      break;
    case 16:
      castElementsToPtr<APInt, int16_t>(elementsA, &dataPtrA);
      castElementsToPtr<APInt, int16_t>(elementsB, &dataPtrB);
      fn.template operator()<int16_t>(
          std::shared_ptr<int16_t>(static_cast<int16_t *>(dataPtrA), free),
          std::shared_ptr<int16_t>(static_cast<int16_t *>(dataPtrB), free));
      break;
    case 8:
      castElementsToPtr<APInt, int8_t>(elementsA, &dataPtrA);
      castElementsToPtr<APInt, int8_t>(elementsB, &dataPtrB);
      fn.template operator()<int8_t>(
          std::shared_ptr<int8_t>(static_cast<int8_t *>(dataPtrA), free),
          std::shared_ptr<int8_t>(static_cast<int8_t *>(dataPtrB), free));
      break;
    default:
      llvm_unreachable("Unsupported integer width. ");
    }
  }
}

template <typename T>
DenseElementsAttr getDenseElementsAttr(Type tp, ArrayRef<int64_t> size, T *data,
                                       int64_t totalSize) {
  return DenseElementsAttr::get(RankedTensorType::get(size, tp),
                                ArrayRef<T>(data, totalSize));
}

template <>
inline DenseElementsAttr getDenseElementsAttr(Type tp, ArrayRef<int64_t> size,
                                              fp16 *data, int64_t totalSize) {
  SmallVector<float> upcastData;
  for (int64_t i = 0; i < totalSize; i++) {
    upcastData.emplace_back(data[i]);
  }
  return DenseElementsAttr::get(RankedTensorType::get(size, tp),
                                ArrayRef<float>(upcastData));
}

enum class NumericTypeID {
  kUnknown,
  kF32,
  kF16,
  kF8,
  kTF32,
  kBF16,
  kI64,
  kI32,
  kI16,
  kI8,
};

template <typename T> struct NumericTypeMap;

template <> struct NumericTypeMap<int64_t> {
  static NumericTypeID const kId = NumericTypeID::kI64;
};

template <> struct NumericTypeMap<int32_t> {
  static NumericTypeID const kId = NumericTypeID::kI32;
};
template <> struct NumericTypeMap<int16_t> {
  static NumericTypeID const kId = NumericTypeID::kI16;
};
template <> struct NumericTypeMap<int8_t> {
  static NumericTypeID const kId = NumericTypeID::kI8;
};

template <> struct NumericTypeMap<float> {
  static NumericTypeID const kId = NumericTypeID::kF32;
};

template <> struct NumericTypeMap<fp16> {
  static NumericTypeID const kId = NumericTypeID::kF16;
};

#ifdef ENABLE_CUDA

template <> struct NumericTypeMap<cutlass::half_t> {
  static NumericTypeID const kId = NumericTypeID::kF16;
};

template <> struct NumericTypeMap<cutlass::bfloat16_t> {
  static NumericTypeID const kId = NumericTypeID::kF16;
};

template <> struct NumericTypeMap<cutlass::tfloat32_t> {
  static NumericTypeID const kId = NumericTypeID::kF16;
};

#endif
} // namespace hands_on_mlir
} // namespace mlir

#endif
