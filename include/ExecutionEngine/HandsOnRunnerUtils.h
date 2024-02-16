#ifndef HANDS_ON_MLIR_EXECUTIONENGINE_RUNNERUTILS_H
#define HANDS_ON_MLIR_EXECUTIONENGINE_RUNNERUTILS_H

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>
#ifdef _WIN32 // Copied from official mlir project
#ifndef HANDS_ON_MLIR_RUNNERUTILS_EXPORT
#ifdef mlir_runner_utils_EXPORTS
// We are building this library
#define HANDS_ON_MLIR_RUNNERUTILS_EXPORT __declspec(dllexport)
#else
// We are using this library
#define HANDS_ON_MLIR_RUNNERUTILS_EXPORT __declspec(dllimport)
#endif // mlir_runner_utils_EXPORTS
#endif // MLIR_RUNNERUTILS_EXPORT
#else
// Non-windows: use visibility attributes.
#define HANDS_ON_MLIR_RUNNERUTILS_EXPORT __attribute__((visibility("default")))
#endif // _WIN32

#include "mlir/ExecutionEngine/RunnerUtils.h"
#include <stddef.h>

struct C_UnrankedMemRefType : UnrankedMemRefType<float> {};

template <class T = float>
auto convertToDynamicMemRefType(int64_t rank, void *dst) {
  UnrankedMemRefType<T> unrankType = {rank, dst};
  DynamicMemRefType<T> dyType(unrankType);
  return dyType;
}

extern "C" {
HANDS_ON_MLIR_RUNNERUTILS_EXPORT void print2DMatrixF32(int64_t rank, void *dst);
HANDS_ON_MLIR_RUNNERUTILS_EXPORT void fill2DRandomMatrixF32(int64_t rank,
                                                            void *dst);
HANDS_ON_MLIR_RUNNERUTILS_EXPORT void fill2DIncMatrixF32(int64_t rank,
                                                         void *dst);
HANDS_ON_MLIR_RUNNERUTILS_EXPORT void validateF32WithRefMatmul(int64_t, void *,
                                                               int64_t, void *,
                                                               int64_t, void *,
                                                               int64_t, void *);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void deallocF32(int64_t rank, void *dst);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT C_UnrankedMemRefType
allocF32(int32_t elementNum);
HANDS_ON_MLIR_RUNNERUTILS_EXPORT
C_UnrankedMemRefType alloc3DMemRefF32(int32_t, int32_t, int32_t);
HANDS_ON_MLIR_RUNNERUTILS_EXPORT C_UnrankedMemRefType
allocByMemRefF32(int64_t rank, void *dst);
HANDS_ON_MLIR_RUNNERUTILS_EXPORT C_UnrankedMemRefType
allocConstantF32(int32_t idx);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void matmulAddF32(int64_t, void *, int64_t,
                                                   void *, int64_t, void *,
                                                   int64_t, void *);
}
#endif
