#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "half.h"
#include "transformer_engine/transformer_engine.h"
#include <cstdint>
#include <cublas.h>

using Status = cutlass::Status;

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
