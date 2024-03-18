#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/library/types.h"
#include "half.h"
#include "transformer_engine/transformer_engine.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

using Status = cutlass::Status;

namespace transformer_engine {
struct SimpleTensor {
  void *dptr;
  std::vector<size_t> shape;
  transformer_engine::DType dtype;

  SimpleTensor(void *dptr, const std::vector<size_t> &shape, DType dtype)
      : dptr(dptr), shape(shape), dtype(dtype) {}
  SimpleTensor() : SimpleTensor(nullptr, {}, DType::kFloat32) {}
};

struct Tensor {
  SimpleTensor data;
  SimpleTensor amax;
  SimpleTensor scale;
  SimpleTensor scale_inv;

  Tensor()
      : data(), amax(nullptr, {1}, DType::kFloat32),
        scale(nullptr, {1}, DType::kFloat32),
        scale_inv(nullptr, {1}, DType::kFloat32) {}
};
} // namespace transformer_engine

#define checkCudaErrors(func)                                                  \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    if (e != cudaSuccess) {                                                    \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));   \
      exit(1);                                                                 \
    }                                                                          \
  }

static const char *_cuBlasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "UNKNOWN_CUBLAS_ERROR";
  }
}

#define checkCuBlasErrors(func)                                                \
  {                                                                            \
    cublasStatus_t e = (func);                                                 \
    if (e != CUBLAS_STATUS_SUCCESS)                                            \
      printf("%s %d CuBlas: %s\n", __FILE__, __LINE__,                         \
             _cuBlasGetErrorEnum(e));                                          \
  }
namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu_kernel {

template <typename T> struct NVTETypeMap;

template <> struct NVTETypeMap<int64_t> {
  static NVTEDType const kType = NVTEDType::kNVTEInt64;
};

template <> struct NVTETypeMap<int32_t> {
  static NVTEDType const kType = NVTEDType::kNVTEInt32;
};

template <> struct NVTETypeMap<float> {
  static NVTEDType const kType = NVTEDType::kNVTEFloat32;
};

template <> struct NVTETypeMap<half> {
  static NVTEDType const kType = NVTEDType::kNVTEFloat16;
};

template <> struct NVTETypeMap<cutlass::half_t> {
  static NVTEDType const kType = NVTEDType::kNVTEFloat16;
};

template <typename T> struct NVTEWrapperDTypeMap;

template <> struct NVTEWrapperDTypeMap<int64_t> {
  static transformer_engine::DType const kType =
      transformer_engine::DType::kInt64;
};

template <> struct NVTEWrapperDTypeMap<int32_t> {
  static transformer_engine::DType const kType =
      transformer_engine::DType::kInt32;
};

template <> struct NVTEWrapperDTypeMap<char> {
  static transformer_engine::DType const kType =
      transformer_engine::DType::kByte;
};

template <> struct NVTEWrapperDTypeMap<float> {
  static transformer_engine::DType const kType =
      transformer_engine::DType::kFloat32;
};

template <> struct NVTEWrapperDTypeMap<half> {
  static transformer_engine::DType const kType =
      transformer_engine::DType::kFloat16;
};

template <> struct NVTEWrapperDTypeMap<cutlass::half_t> {
  static transformer_engine::DType const kType =
      transformer_engine::DType::kFloat16;
};

inline size_t getNVTEWrapperDTypeSize(transformer_engine::DType dtype) {
  using DType = transformer_engine::DType;
  switch (dtype) {
  case DType::kInt64:
    return sizeof(int64_t);
    break;
  case DType::kInt32:
  case DType::kFloat32:
    return sizeof(int32_t);
    break;
  case DType::kFloat16:
  case DType::kBFloat16:
    return sizeof(int16_t);
    break;
  case DType::kFloat8E4M3:
  case DType::kFloat8E5M2:
    return sizeof(int8_t);
    break;
  case DType::kByte:
    return sizeof(char);
    break;
  default:
    llvm_unreachable("Not ok");
  }
}

template <typename T> static std::shared_ptr<T> getZeroPointer(size_t size) {
  static std::shared_ptr<T> ptr;
  static size_t allocedSize = 0;

  if (allocedSize < size) {
    T *allocPtr;
    allocedSize = size;
    checkCudaErrors(cudaMalloc(&allocPtr, sizeof(T) * allocedSize));
    checkCudaErrors(cudaMemset(allocPtr, 0, sizeof(T) * allocedSize));
    ptr.reset(allocPtr, cudaFree);
  }

  return ptr;
}

template <typename T> static std::shared_ptr<T> getOnePointer(size_t size) {
  static std::shared_ptr<T> ptr;
  static size_t allocedSize = 0;

  if (allocedSize < size) {
    T *allocPtr;
    allocedSize = size;
    checkCudaErrors(cudaMalloc(&allocPtr, sizeof(T) * allocedSize));
    auto thrustPtr = thrust::device_pointer_cast(allocPtr);
    thrust::fill(thrustPtr, thrustPtr + size, T(1));
    ptr.reset(allocPtr, cudaFree);
  }

  return ptr;
}

template <typename T = char>
static std::shared_ptr<T> getDummyPointer(size_t size) {
  static std::shared_ptr<T> ptr;
  static size_t allocedSize = 0;

  if (allocedSize < size) {
    T *allocPtr;
    allocedSize = size;
    checkCudaErrors(cudaMalloc(&allocPtr, sizeof(T) * allocedSize));
    auto deleter = [](T *data) { cudaFree(data); };
    ptr.reset(allocPtr, deleter);
  }

  return ptr;
}

static auto getMulitProcessorCount() {
  static int mpCount = -1;

  if (mpCount == -1) {
    int device_id;
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDevice(&device_id));
    cudaGetDeviceProperties(&prop, device_id);
    mpCount = prop.multiProcessorCount;
  }

  return mpCount;
}

using LayoutTypeID = cutlass::library::LayoutTypeID;
using NumericTypeID = cutlass::library::NumericTypeID;

template <typename T> struct LayoutMap;

template <> struct LayoutMap<cutlass::layout::ColumnMajor> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajor;
};

template <> struct LayoutMap<cutlass::layout::RowMajor> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajor;
};

template <> struct LayoutMap<cutlass::layout::ColumnMajorInterleaved<2>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK2;
};

template <> struct LayoutMap<cutlass::layout::RowMajorInterleaved<2>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK2;
};

template <> struct LayoutMap<cutlass::layout::ColumnMajorInterleaved<4>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK4;
};

template <> struct LayoutMap<cutlass::layout::RowMajorInterleaved<4>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK4;
};

template <> struct LayoutMap<cutlass::layout::ColumnMajorInterleaved<16>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK16;
};

template <> struct LayoutMap<cutlass::layout::RowMajorInterleaved<16>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK16;
};

template <> struct LayoutMap<cutlass::layout::ColumnMajorInterleaved<32>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK32;
};

template <> struct LayoutMap<cutlass::layout::RowMajorInterleaved<32>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK32;
};

template <> struct LayoutMap<cutlass::layout::ColumnMajorInterleaved<64>> {
  static LayoutTypeID const kId = LayoutTypeID::kColumnMajorInterleavedK64;
};

template <> struct LayoutMap<cutlass::layout::RowMajorInterleaved<64>> {
  static LayoutTypeID const kId = LayoutTypeID::kRowMajorInterleavedK64;
};

template <typename T> struct NumericTypeMap;

template <> struct NumericTypeMap<void> {
  static NumericTypeID const kId = NumericTypeID::kVoid;
};

template <> struct NumericTypeMap<cutlass::uint1b_t> {
  static NumericTypeID const kId = NumericTypeID::kB1;
};

template <> struct NumericTypeMap<cutlass::int4b_t> {
  static NumericTypeID const kId = NumericTypeID::kS4;
};

template <> struct NumericTypeMap<int8_t> {
  static NumericTypeID const kId = NumericTypeID::kS8;
};

template <> struct NumericTypeMap<int16_t> {
  static NumericTypeID const kId = NumericTypeID::kS16;
};

template <> struct NumericTypeMap<int32_t> {
  static NumericTypeID const kId = NumericTypeID::kS32;
};

template <> struct NumericTypeMap<int64_t> {
  static NumericTypeID const kId = NumericTypeID::kS64;
};

template <> struct NumericTypeMap<cutlass::uint4b_t> {
  static NumericTypeID const kId = NumericTypeID::kU4;
};

template <> struct NumericTypeMap<uint8_t> {
  static NumericTypeID const kId = NumericTypeID::kU8;
};

template <> struct NumericTypeMap<cutlass::float_e4m3_t> {
  static NumericTypeID const kId = NumericTypeID::kFE4M3;
};

template <> struct NumericTypeMap<cutlass::float_e5m2_t> {
  static NumericTypeID const kId = NumericTypeID::kFE5M2;
};

template <> struct NumericTypeMap<uint16_t> {
  static NumericTypeID const kId = NumericTypeID::kU16;
};

template <> struct NumericTypeMap<uint32_t> {
  static NumericTypeID const kId = NumericTypeID::kU32;
};

template <> struct NumericTypeMap<uint64_t> {
  static NumericTypeID const kId = NumericTypeID::kU64;
};

template <> struct NumericTypeMap<cutlass::half_t> {
  static NumericTypeID const kId = NumericTypeID::kF16;
};

template <> struct NumericTypeMap<float> {
  static NumericTypeID const kId = NumericTypeID::kF32;
};

template <> struct NumericTypeMap<double> {
  static NumericTypeID const kId = NumericTypeID::kF64;
};

template <> struct NumericTypeMap<cutlass::bfloat16_t> {
  static NumericTypeID const kId = NumericTypeID::kBF16;
};

template <> struct NumericTypeMap<cutlass::tfloat32_t> {
  static NumericTypeID const kId = NumericTypeID::kTF32;
};

} // namespace homnvgpu_kernel
} // namespace hands_on_mlir
} // namespace mlir
