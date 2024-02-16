#ifndef HANDS_ON_MLIR_WEIGHTSENGINE_UTILS_H_
#define HANDS_ON_MLIR_WEIGHTSENGINE_UTILS_H_

#include "cutlass/bfloat16.h"
#include "cutlass/half.h"
#include "cutlass/tfloat32.h"
#include "half.hpp"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/VersionTuple.h"
#include <cstdint>

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

template <> inline float convertToT0(APFloat data) {
  return data.convertToFloat();
}

template <> inline half_float::half convertToT0(APFloat data) {
  return half_float::half(data.convertToFloat());
}

template <class T> T convertToT0(APInt data) { return data.getLimitedValue(); }

template <class T, class T0>
void castElementsToPtr(ElementsAttr &element, void **ptrptr) {
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

template <> struct NumericTypeMap<half_float::half> {
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
