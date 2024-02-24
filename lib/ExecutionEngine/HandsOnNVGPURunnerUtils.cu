#include "ExecutionEngine/HandsOnNVGPURunnerUtils.h"
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/BertAttentionRunner.h"
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
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern "C" {

void cutlassGemmF32(int64_t rankA, void *dstA, int64_t rankB, void *dstB,
                    int64_t rankC, void *dstC, int64_t rankD, void *dstD,
                    float alpha, float beta) {

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

  std::cout << cutlass::cutlassGetStatusString(status) << std::endl;

  assert(status == cutlass::Status::kSuccess);
}

void cutlassGemmF16(int64_t rankA, void *dstA, int64_t rankB, void *dstB,
                    int64_t rankC, void *dstC, int64_t rankD, void *dstD,
                    float alpha, float beta) {

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

  std::cout << cutlass::cutlassGetStatusString(status) << std::endl;

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

void cutlassLayernormGemmF32(int64_t rankA, void *dstA, int64_t rankB,
                             void *dstB, int64_t rankC, void *dstC,
                             int64_t rankD, void *dstD, int64_t rankVar,
                             void *dstVar, int64_t rankMean, void *dstMean,
                             float alpha, float beta, float eps,
                             int64_t activation) {}

C_UnrankedMemRefType allocConstantNVGPUF32(int32_t idx) {
  auto haha = C_UnrankedMemRefType();
  // So stupid...
  std::ifstream file(std::filesystem::path(__FILE__).parent_path().string() +
                     std::string("/../../examples/torch/linear/") +
                     std::to_string(idx) + ".txt");
  std::vector<int> v;
  int a;
  std::string line;
  getline(file, line);
  std::stringstream ss(line);
  while (ss >> a) {
    v.push_back(a);
  }
  haha.rank = v.size();
  assert(haha.rank == 3);
  haha.descriptor = malloc(
      sizeof(StridedMemRefType<float, 3>)); // MLIR will delete this ptr for us.
  auto des = static_cast<StridedMemRefType<float, 3> *>(haha.descriptor);
  int32_t stride = 1;
  for (size_t i = 0; i < v.size(); i++) {
    des->sizes[i] = v[i];
    des->strides[2 - i] = stride;
    stride *= v[2 - i];
  }
  checkCudaErrors(cudaMalloc(&(des->data), sizeof(float) * stride));
  std::cout << "Allocate const on cuda: " << des->data << std::endl;
  float *host_data = new float[stride];
  for (int i = 0; i < stride; i++) {
    file >> host_data[i];
  }
  checkCudaErrors(cudaMemcpy(des->data, host_data, sizeof(float) * stride,
                             cudaMemcpyHostToDevice));
  des->basePtr = des->data;
  des->offset = 0;
  delete[] host_data;
  return haha;
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
  auto returnMemRef = C_UnrankedMemRefType();
  returnMemRef.rank = 3;
  returnMemRef.descriptor = malloc(sizeof(StridedMemRefType<float, 3>));
  auto des =
      static_cast<StridedMemRefType<float, 3> *>(returnMemRef.descriptor);
  checkCudaErrors(cudaMalloc(&(des->data), sizeof(float) * a * b * c));
  std::cout << "Allocate 3d tensor on cuda: " << des->data << std::endl;
  std::cout << "Size: " << a << " " << b << " " << c << std::endl;
  des->basePtr = des->data;
  des->offset = 0;
  des->sizes[0] = a;
  des->sizes[1] = b;
  des->sizes[2] = c;
  des->strides[0] = b * c;
  des->strides[1] = c;
  des->strides[2] = 1;
  return returnMemRef;
}

void deallocNVGPUF32(int64_t rank, void *dst) {
  auto memRef = convertToDynamicMemRefType(rank, dst);
  cudaFree(memRef.data);
}
}
