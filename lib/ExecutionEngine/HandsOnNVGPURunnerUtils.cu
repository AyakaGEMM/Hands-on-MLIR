#include "ExecutionEngine/HandsOnNVGPURunnerUtils.h"
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/BertAttentionRunner.h"
#include "NVGPUKernels/CuSeqLen.h"
#include "NVGPUKernels/ElementwiseRunner.h"
#include "NVGPUKernels/GatherRunner.h"
#include "NVGPUKernels/GemmRunner.h"
#include "NVGPUKernels/Layernorm.h"
#include "NVGPUKernels/Utils.h"
#include "NVGPUKernels/gemm_with_epilogue_visitor.h"
#include "cute/numeric/int.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/kernel/gemm_universal_with_visitor.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <complex.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

allocFnType nvgpuAllocer = [](void **ptr, size_t size) {
  checkCudaErrors(cudaMalloc(ptr, size));
  std::cout << "Allocate 3d tensor on cuda: " << *ptr << std::endl;
  std::cout << "Size: " << size << std::endl;
};

extern "C" {

void cutlassGemmF32(int64_t rankA, void *dstA, bool transa, int64_t rankB,
                    void *dstB, bool transb, int64_t rankC, void *dstC,
                    int64_t rankD, void *dstD, int64_t activation, float alpha,
                    float beta) {

  // Ideally, we should use manifest with generated template here.
  using RowMajor = cutlass::layout::RowMajor;
  using CutlassGemm =
      cutlass::gemm::device::Gemm<float,     // Data-type of A matrix
                                  RowMajor,  // Layout of A matrix
                                  float,     // Data-type of B matrix
                                  RowMajor,  // Layout of B matrix
                                  float,     // Data-type of C matrix
                                  RowMajor>; // Layout of C matrix

  mlir::hands_on_mlir::GemmOperationRunner<CutlassGemm> gemm;

  auto status =
      gemm.run(rankA, dstA, rankB, dstB, rankC, dstC, rankD, dstD, alpha, beta);

  assert(status == cutlass::Status::kSuccess);
}

void cutlassGemmF16(int64_t rankA, void *dstA, bool transa, int64_t rankB,
                    void *dstB, bool transb, int64_t rankC, void *dstC,
                    int64_t rankD, void *dstD, int64_t activation, float alpha,
                    float beta) {

  // Ideally, we should use manifest with generated template here.
  using RowMajor = cutlass::layout::RowMajor;
  using CutlassGemm =
      cutlass::gemm::device::Gemm<cutlass::half_t, // Data-type of A matrix
                                  RowMajor,        // Layout of A matrix
                                  cutlass::half_t, // Data-type of B matrix
                                  RowMajor,        // Layout of B matrix
                                  cutlass::half_t, // Data-type of C matrix
                                  RowMajor>;       // Layout of C matrix

  mlir::hands_on_mlir::GemmOperationRunner<CutlassGemm> gemm;

  auto status =
      gemm.run(rankA, dstA, rankB, dstB, rankC, dstC, rankD, dstD, alpha, beta);

  assert(status == cutlass::Status::kSuccess);
}

void cutlassGemmWithVarMeanF16(int64_t rankA, void *dstA, int64_t rankB,
                               void *dstB, int64_t rankC, void *dstC,
                               int64_t rankD, void *dstD, int64_t rankVar,
                               void *dstVar, int64_t rankMean, void *dstMean,
                               float alpha, float beta, int64_t activation,
                               float eps) {

  using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value,
      cutlass::half_t, cutlass::half_t>;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using DefaultGemmKernel =
      typename cutlass::gemm::kernel::DefaultGemmUniversal<
          cutlass::half_t, cutlass::layout::RowMajor,
          cutlass::ComplexTransform::kNone,
          128 / cutlass::sizeof_bits<cutlass::half_t>::value, cutlass::half_t,
          cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone,
          128 / cutlass::sizeof_bits<cutlass::half_t>::value, cutlass::half_t,
          cutlass::layout::RowMajor, cutlass::half_t,
          cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape,
          WarpShape, InstructionShape, EpilogueFunctorOp,
          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2,
          typename cutlass::gemm::device::DefaultGemmConfiguration<
              cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
              cutlass::half_t, cutlass::half_t, cutlass::half_t,
              cutlass::half_t>::Operator>::GemmKernel;

  using EpilogueVisitor = mlir::hands_on_mlir::EpilogueVisitorLayerNorm<
      ThreadblockShape, DefaultGemmKernel::kThreadCount,
      DefaultGemmKernel::Epilogue::OutputTileIterator,
      DefaultGemmKernel::Epilogue::AccumulatorFragmentIterator::AccumulatorTile,
      cutlass::half_t, cutlass::half_t, cutlass::half_t, float,
      EpilogueFunctorOp>;

  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogue<
          EpilogueVisitor, typename DefaultGemmKernel::Epilogue>::Epilogue;

  using Gemm = cutlass::gemm::kernel::GemmWithEpilogueVisitorFromExample<
      DefaultGemmKernel::Mma, Epilogue, DefaultGemmKernel::ThreadblockSwizzle>;

  mlir::hands_on_mlir::GemmOperationWithVarMeanRunner<Gemm> gemm;

  auto status = gemm.run(rankA, dstA, rankB, dstB, rankC, dstC, rankD, dstD,
                         rankVar, dstVar, rankMean, dstMean, alpha, beta, eps);

  assert(status == cutlass::Status::kSuccess);
}

void nvteGemmF16(int64_t rankA, void *dstA, bool transa, int64_t rankB,
                 void *dstB, bool transb, int64_t rankC, void *dstC,
                 int64_t rankD, void *dstD, int64_t activation, float alpha,
                 float beta) {
  mlir::hands_on_mlir::GemmNVTERunner<half> gemm;

  auto status = gemm.run(rankA, dstA, transa, rankB, dstB, transb, rankC, dstC,
                         rankD, dstD, activation, alpha, beta);

  assert(status == cutlass::Status::kSuccess);
}

void cutlassLayernormGemmF32(int64_t rankA, void *dstA, int64_t rankB,
                             void *dstB, int64_t rankC, void *dstC,
                             int64_t rankD, void *dstD, int64_t rankVar,
                             void *dstVar, int64_t rankMean, void *dstMean,
                             float alpha, float beta, float eps,
                             int64_t activation) {}

C_UnrankedMemRefType allocConstantNVGPUF32(int32_t idx) {

  std::ifstream file(std::filesystem::path(__FILE__).parent_path().string() +
                     std::string("/../../examples/torch/linear/") +
                     std::to_string(idx) + ".txt");

  std::vector<int64_t> v;
  int64_t a;
  std::string line;
  getline(file, line);
  std::stringstream ss(line);
  while (ss >> a) {
    v.push_back(a);
  }

  assert(v.size() == 3);

  auto res = allocHelper<float, 3>(v, nvgpuAllocer);
  auto des = static_cast<StridedMemRefType<float, 3> *>(res.descriptor);
  auto totalSize = std::accumulate(v.begin(), v.end(), 1, std::multiplies<>());
  auto host_data = new float[totalSize];
  float tmp;
  for (int i = 0; i < totalSize; i++) {
    file >> tmp;
    host_data[i] = tmp;
  }

  std::cout << totalSize << " " << des->data << std::endl;

  checkCudaErrors(cudaMemcpy(des->data, host_data, sizeof(float) * totalSize,
                             cudaMemcpyHostToDevice));
  delete[] host_data;
  return res;
}

C_UnrankedMemRefType allocConstantNVGPUF16(int32_t idx) {

  std::ifstream file(std::filesystem::path(__FILE__).parent_path().string() +
                     std::string("/../../examples/torch/linear/") +
                     std::to_string(idx) + ".txt");

  std::vector<int64_t> v;
  int64_t a;
  std::string line;
  getline(file, line);
  std::stringstream ss(line);
  while (ss >> a) {
    v.push_back(a);
  }

  assert(v.size() == 3);

  auto res = allocHelper<half, 3>(v, nvgpuAllocer);
  auto des = static_cast<StridedMemRefType<half, 3> *>(res.descriptor);
  auto totalSize = std::accumulate(v.begin(), v.end(), 1, std::multiplies<>());
  auto host_data = new half[totalSize];
  float tmp;
  for (int i = 0; i < totalSize; i++) {
    file >> tmp;
    host_data[i] = tmp;
  }

  std::cout << totalSize << " " << des->data << std::endl;

  checkCudaErrors(cudaMemcpy(des->data, host_data, sizeof(half) * totalSize,
                             cudaMemcpyHostToDevice));
  delete[] host_data;
  return res;
}

void nvteLayernormF32(int64_t rankA, void *dstA, float eps) {
  mlir::hands_on_mlir::LayernormRunner<float> lnRunner;
  lnRunner.run(rankA, dstA, eps);
}

void nvteLayernormF16(int64_t rankA, void *dstA, float eps) {
  mlir::hands_on_mlir::LayernormRunner<half> lnRunner;
  lnRunner.run(rankA, dstA, eps);
}

void nvteBertAttentionF32(int64_t rankA, void *dstA, int64_t rankSeqlen,
                          void *dstSeqlen, int64_t rankOut, void *dstOut,
                          float scale, int64_t headNum) {
  mlir::hands_on_mlir::BertAttentionRunner<float> bertAttnRunner;
  bertAttnRunner.run(rankA, dstA, rankSeqlen, dstSeqlen, rankOut, dstOut, scale,
                     headNum);
}

void nvteBertAttentionF16(int64_t rankA, void *dstA, int64_t rankSeqlen,
                          void *dstSeqlen, int64_t rankOut, void *dstOut,
                          float scale, int64_t headNum) {
  mlir::hands_on_mlir::BertAttentionRunner<half> bertAttnRunner;
  bertAttnRunner.run(rankA, dstA, rankSeqlen, dstSeqlen, rankOut, dstOut, scale,
                     headNum);
}

C_UnrankedMemRefType alloc3DMemRefNVGPUF32(int32_t a, int32_t b, int32_t c) {
  return allocHelper<float, 3>({a, b, c}, nvgpuAllocer);
}

C_UnrankedMemRefType alloc3DMemRefNVGPUF16(int32_t a, int32_t b, int32_t c) {
  return allocHelper<half, 3>({a, b, c}, nvgpuAllocer);
}

C_UnrankedMemRefType alloc1DMemRefNVGPUI32(int32_t a) {
  return allocHelper<int32_t, 1>({a}, nvgpuAllocer);
}

void deallocNVGPUF32(int64_t rank, void *dst) {
  auto memRef = convertToDynamicMemRefType(rank, dst);
  cudaFree(memRef.data);
}

void deallocNVGPUF16(int64_t rank, void *dst) {
  auto memRef = convertToDynamicMemRefType<half>(rank, dst);
  cudaFree(memRef.data);
}

void deallocNVGPUI32(int64_t rank, void *dst) {
  auto memRef = convertToDynamicMemRefType<int32_t>(rank, dst);
  cudaFree(memRef.data);
}

thrustElementwiseDEF(Add, F32, float);
thrustElementwiseDEF(Sub, F32, float);
thrustElementwiseDEF(Mul, F32, float);
thrustElementwiseDEF(Div, F32, float);

thrustElementwiseDEF(Add, F16, half);
thrustElementwiseDEF(Sub, F16, half);
thrustElementwiseDEF(Mul, F16, half);
thrustElementwiseDEF(Div, F16, half);

thrustGatherDEF(F32, float);
thrustGatherDEF(F16, half);

thrustCuSeqLenDEF(I64, int64_t);
thrustCuSeqLenDEF(I32, int32_t);
}
