#ifndef HANDS_ON_MLIR_NVGPU_RUNNER_UTILS_H
#define HANDS_ON_MLIR_NVGPU_RUNNER_UTILS_H

#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/Utils.h"

extern "C" {
HANDS_ON_MLIR_RUNNERUTILS_EXPORT void cutlassGemmF32(int64_t rankA, void *dstA,
                                                     int64_t rankB, void *dstB,
                                                     int64_t rankC, void *dstC,
                                                     int64_t rankD, void *dstD,
                                                     float alpha, float beta);
HANDS_ON_MLIR_RUNNERUTILS_EXPORT C_UnrankedMemRefType
allocConstantNVGPUF32(int32_t idx);
HANDS_ON_MLIR_RUNNERUTILS_EXPORT
C_UnrankedMemRefType alloc3DMemRefNVGPUF32(int32_t, int32_t, int32_t);
HANDS_ON_MLIR_RUNNERUTILS_EXPORT void deallocNVGPUF32(int64_t rank, void *dst);
}

#endif
