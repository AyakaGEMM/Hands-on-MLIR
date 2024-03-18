#ifndef HANDS_ON_MLIR_NVGPU_RUNNER_UTILS_H
#define HANDS_ON_MLIR_NVGPU_RUNNER_UTILS_H

#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/Utils.h"
#include <cstdint>

extern allocFnType nvgpuAllocer;

#define thrustElementwiseDECL(op, suffix)                                      \
  HANDS_ON_MLIR_RUNNERUTILS_EXPORT void thrustElementwise##op##suffix(         \
      int64_t, void *, int64_t, void *, int64_t, void *);

#define thrustElementwiseDEF(op, suffix, type)                                 \
  void thrustElementwise##op##suffix(int64_t rankA, void *dstA, int64_t rankB, \
                                     void *dstB, int64_t rankOut,              \
                                     void *dstOut) {                           \
    using namespace mlir::hands_on_mlir::homnvgpu_kernel;                      \
    ElementwiseRunner<op<type>> runner;                                        \
    runner.run(rankA, dstA, rankB, dstB, rankOut, dstOut);                     \
  }

#define thrustGatherDECL(suffix)                                               \
  HANDS_ON_MLIR_RUNNERUTILS_EXPORT void thrustGather##suffix(                  \
      int64_t, void *, int64_t, void *, int64_t, void *);

#define thrustGatherDEF(suffix, type)                                          \
  void thrustGather##suffix(int64_t rankIndices, void *desIndices,             \
                            int64_t rankValue, void *desValue,                 \
                            int64_t rankOut, void *desOut) {                   \
    using namespace mlir::hands_on_mlir::homnvgpu_kernel;                      \
    GatherRunner<type> runner;                                                 \
    runner.run(rankIndices, desIndices, rankValue, desValue, rankOut, desOut); \
  }

#define thrustElementwiseDEF(op, suffix, type)                                 \
  void thrustElementwise##op##suffix(int64_t rankA, void *dstA, int64_t rankB, \
                                     void *dstB, int64_t rankOut,              \
                                     void *dstOut) {                           \
    using namespace mlir::hands_on_mlir::homnvgpu_kernel;                      \
    ElementwiseRunner<op<type>> runner;                                        \
    runner.run(rankA, dstA, rankB, dstB, rankOut, dstOut);                     \
  }

#define thrustCuSeqLenDECL(suffix)                                             \
  HANDS_ON_MLIR_RUNNERUTILS_EXPORT void thrustCuSeqLen##suffix(                \
      int64_t, void *, int64_t, void *);

#define thrustCuSeqLenDEF(suffix, type)                                        \
  void thrustCuSeqLen##suffix(int64_t rankInput, void *desInput,               \
                              int64_t rankOut, void *desOut) {                 \
    using namespace mlir::hands_on_mlir::homnvgpu_kernel;                      \
    CuSeqLenRunner<type> runner;                                               \
    runner.run(rankInput, desInput, rankOut, desOut);                          \
  }

extern "C" {
HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
cutlassGemmF32(int64_t rankA, void *dstA, bool transa, int64_t rankB,
               void *dstB, bool transb, int64_t rankC, void *dstC,
               int64_t rankD, void *dstD, int64_t activation, float alpha,
               float beta);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
cutlassGemmF16(int64_t rankA, void *dstA, bool transa, int64_t rankB,
               void *dstB, bool transb, int64_t rankC, void *dstC,
               int64_t rankD, void *dstD, int64_t activation, float alpha,
               float beta);

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
                        void *dstMean, float alpha, float beta,
                        int64_t activation);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
cutlassLayernormGemmF16(int64_t rankA, void *dstA, int64_t rankB, void *dstB,
                        int64_t rankC, void *dstC, int64_t rankD, void *dstD,
                        int64_t rankVar, void *dstVar, int64_t rankMean,
                        void *dstMean, float alpha, float beta,
                        int64_t activation);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT
void nvteGemmF16(int64_t rankA, void *dstA, bool transa, int64_t rankB,
                 void *dstB, bool transb, int64_t rankC, void *dstC,
                 int64_t rankD, void *dstD, int64_t activation, float alpha,
                 float beta, int32_t, int32_t);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
nvteLayernormF32(int64_t rankA, void *dstA, float eps = 1e-6);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
nvteLayernormF16(int64_t rankA, void *dstA, float eps = 1e-6);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
nvteBertAttentionF32(int64_t rankA, void *dstA, int64_t rankSeqlen,
                     void *dstSeqlen, int64_t rankOut, void *dstOut,
                     float scale, int64_t headNum);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
nvteBertAttentionF16(int64_t rankA, void *dstA, int64_t rankSeqlen,
                     void *dstSeqlen, int64_t rankOut, void *dstOut,
                     float scale, int64_t headNum);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT C_UnrankedMemRefType
allocConstantNVGPUF32(int32_t idx);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT C_UnrankedMemRefType
allocConstantNVGPUF16(int32_t idx);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT
C_UnrankedMemRefType alloc3DMemRefNVGPUF32(int32_t, int32_t, int32_t);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT
C_UnrankedMemRefType alloc3DMemRefNVGPUF16(int32_t, int32_t, int32_t);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT
C_UnrankedMemRefType alloc1DMemRefNVGPUI32(int32_t);

HANDS_ON_MLIR_RUNNERUTILS_EXPORT void deallocNVGPUF32(int64_t rank, void *dst);
HANDS_ON_MLIR_RUNNERUTILS_EXPORT void deallocNVGPUF16(int64_t rank, void *dst);
HANDS_ON_MLIR_RUNNERUTILS_EXPORT void deallocNVGPUI32(int64_t rank, void *dst);

thrustElementwiseDECL(Add, F32);
thrustElementwiseDECL(Sub, F32);
thrustElementwiseDECL(Mul, F32);
thrustElementwiseDECL(Div, F32);

thrustElementwiseDECL(Add, F16);
thrustElementwiseDECL(Sub, F16);
thrustElementwiseDECL(Mul, F16);
thrustElementwiseDECL(Div, F16);

thrustGatherDECL(F32);
thrustGatherDECL(F16);

thrustCuSeqLenDECL(I32);
thrustCuSeqLenDECL(I64);
}

#endif
