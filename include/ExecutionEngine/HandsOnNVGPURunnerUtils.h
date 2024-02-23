#ifndef HANDS_ON_MLIR_NVGPU_RUNNER_UTILS_H
#define HANDS_ON_MLIR_NVGPU_RUNNER_UTILS_H

#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/Utils.h"
#include <cstdint>

extern "C" {
HANDS_ON_MLIR_RUNNERUTILS_EXPORT void cutlassGemmF32(int64_t rankA, void *dstA,
                                                     int64_t rankB, void *dstB,
                                                     int64_t rankC, void *dstC,
                                                     int64_t rankD, void *dstD,
                                                     float alpha, float beta);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void cutlassGemmF16(int64_t rankA, void *dstA,
                                                     int64_t rankB, void *dstB,
                                                     int64_t rankC, void *dstC,
                                                     int64_t rankD, void *dstD,
                                                     float alpha, float beta);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
cutlassGemmWithVarMeanF16(int64_t rankA, void *dstA, int64_t rankB, void *dstB,
                          int64_t rankC, void *dstC, int64_t rankD, void *dstD,
                          int64_t rankVar, void *dstVar, int64_t rankMean,
                          void *dstMean, float alpha, float beta,
                          int64_t activation, float eps);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
cutlassLayernormGemmF32(int64_t rankA, void *dstA, int64_t rankB, void *dstB,
                        int64_t rankC, void *dstC, int64_t rankD, void *dstD,
                        int64_t rankVar, void *dstVar, int64_t rankMean,
                        void *dstMean, float alpha, float beta, float eps,
                        int64_t activation);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
cutlassLayernormGemmF16(int64_t rankA, void *dstA, int64_t rankB, void *dstB,
                        int64_t rankC, void *dstC, int64_t rankD, void *dstD,
                        int64_t rankVar, void *dstVar, int64_t rankMean,
                        void *dstMean, float alpha, float beta, float eps,
                        int64_t activation);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
nvteLayernormF32(int64_t rankA, void *dstA, float eps = 1e-6);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
nvteLayernormF16(int64_t rankA, void *dstA, float eps = 1e-6);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT C_UnrankedMemRefType
allocConstantNVGPUF32(int32_t idx);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT
C_UnrankedMemRefType alloc3DMemRefNVGPUF32(int32_t, int32_t, int32_t);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void deallocNVGPUF32(int64_t rank, void *dst);
}

#endif