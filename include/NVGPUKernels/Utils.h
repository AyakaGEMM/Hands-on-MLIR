#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "half.h"
#include "transformer_engine/transformer_engine.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cublas.h>
#include <memory>

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

template <typename T = char>
static std::shared_ptr<T> getDummyPointer(size_t size) {
  static std::shared_ptr<T> ptr;
  static size_t allocedSize = 0;

  if (allocedSize < size) {
    T *allocPtr;
    allocedSize = size;
    checkCudaErrors(cudaMalloc(&allocPtr, sizeof(T) * allocedSize));
    ptr.reset(allocPtr, cudaFree);
  }

  return ptr;
}
