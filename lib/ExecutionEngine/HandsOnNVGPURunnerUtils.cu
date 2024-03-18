#include "ExecutionEngine/HandsOnNVGPURunnerUtils.h"
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/BertAttentionRunner.h"
#include "NVGPUKernels/CuSeqLen.h"
#include "NVGPUKernels/ElementwiseRunner.h"
#include "NVGPUKernels/GatherRunner.h"
#include "NVGPUKernels/GemmManifest.h"
#include "NVGPUKernels/GemmRunner.h"
#include "NVGPUKernels/Layernorm.h"
#include "NVGPUKernels/LayernormGemmRunner.h"
#include "NVGPUKernels/Utils.h"
#include "NVGPUKernels/gemm_with_epilogue_visitor.h"
#include "cublas_v2.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
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

template <typename T>
static bool checkSame(int64_t rankA, void *desA, int64_t rankB, void *desB) {
  if (rankA != rankB) {
    std::cout << "Shape not same" << rankA << " " << rankB << std::endl;
    return false;
  }

  auto a = convertToDynamicMemRefType<T>(rankA, desA);
  auto b = convertToDynamicMemRefType<T>(rankB, desB);

  for (int i = 0; i < rankA; i++) {
    if (a.sizes[i] != b.sizes[i]) {
      return false;
    }

    if (a.strides[i] != b.strides[i]) {
      return false;
    }
  }

  auto totalSize =
      std::accumulate(a.sizes, a.sizes + rankA, 1, std::multiplies<>());

  for (int i = 0; i < totalSize; i++) {
    if (a.data[i] != b.data[i]) {
      return false;
    }
  }

  return true;
}

extern "C" {

void cutlassGemmF32(int64_t rankA, void *dstA, bool transa, int64_t rankB,
                    void *dstB, bool transb, int64_t rankC, void *dstC,
                    int64_t rankD, void *dstD, int64_t activation, float alpha,
                    float beta) {
  using namespace mlir::hands_on_mlir::homnvgpu_kernel;

  // Ideally, we should use manifest with generated template here.
  using RowMajor = cutlass::layout::RowMajor;
  using CutlassGemm =
      cutlass::gemm::device::Gemm<float,     // Data-type of A matrix
                                  RowMajor,  // Layout of A matrix
                                  float,     // Data-type of B matrix
                                  RowMajor,  // Layout of B matrix
                                  float,     // Data-type of C matrix
                                  RowMajor>; // Layout of C matrix

  // GemmOperationRunner<CutlassGemm> gemm;

  // auto status = gemm.run(rankA, dstA, rankB, dstB, rankC, dstC, rankD, dstD,
  //                        alpha, beta, 1);

  // assert(status == cutlass::Status::kSuccess);
}

void cutlassGemmF16(int64_t rankA, void *dstA, bool transa, int64_t rankB,
                    void *dstB, bool transb, int64_t rankC, void *dstC,
                    int64_t rankD, void *dstD, int64_t activation, float alpha,
                    float beta, int32_t gemmNum, int32_t splitKFactor) {
  using namespace mlir::hands_on_mlir::homnvgpu_kernel;

  // auto gemm = manifest[gemmNum - 1].get();

  // auto status = gemm->run(rankA, dstA, rankB, dstB, rankC, dstC, rankD, dstD,
  //                         alpha, beta, 1);

  // assert(status == cutlass::Status::kSuccess);
  cublasHandle_t handle;
  checkCuBlasErrors(cublasCreate(&handle));

  auto a = static_cast<StridedMemRefType<half, 3> *>(dstA);
  auto b = static_cast<StridedMemRefType<half, 3> *>(dstB);
  auto c = static_cast<StridedMemRefType<half, 3> *>(dstC);
  auto d = allocHelper<half, 3>({c->sizes[0], c->sizes[1], c->sizes[2]},
                                nvgpuAllocer);
  auto desD = static_cast<StridedMemRefType<half, 3> *>(d.descriptor);
  auto trueD = static_cast<StridedMemRefType<half, 3> *>(dstD);

  cudaMemcpy(desD->data, c->data,
             sizeof(half) * c->sizes[0] * c->sizes[1] * c->sizes[2],
             cudaMemcpyDeviceToDevice);

  half al = alpha, be = beta;

  checkCudaErrors(cudaStreamSynchronize(nullptr));

  cublasHgemm(handle, transa ? CUBLAS_OP_T : CUBLAS_OP_N,
              transb ? CUBLAS_OP_T : CUBLAS_OP_N, b->sizes[2],
              a->sizes[0] * a->sizes[1], a->sizes[2], &al, b->data, b->sizes[2],
              a->data, a->sizes[2], &be, desD->data, desD->sizes[2]);

  checkCudaErrors(cudaStreamSynchronize(nullptr));

  // checkSame<half>(3, d.descriptor, 3, dstD);

  cudaMemcpy(desD->data, trueD->data,
             sizeof(half) * c->sizes[0] * c->sizes[1] * c->sizes[2],
             cudaMemcpyDeviceToDevice);

  cudaFree(desD->data);
  free(desD);

  cublasDestroy(handle);
}

void cutlassGemmWithVarMeanF16(int64_t rankA, void *dstA, int64_t rankB,
                               void *dstB, int64_t rankC, void *dstC,
                               int64_t rankD, void *dstD, int64_t rankVar,
                               void *dstVar, int64_t rankMean, void *dstMean,
                               float alpha, float beta, int64_t activation,
                               float eps) {
  using namespace mlir::hands_on_mlir::homnvgpu_kernel;

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
          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3,
          typename cutlass::gemm::device::DefaultGemmConfiguration<
              cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
              cutlass::half_t, cutlass::half_t, cutlass::half_t,
              cutlass::half_t>::Operator>::GemmKernel;

  using EpilogueVisitor = EpilogueVisitorLayerNorm<
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

  GemmOperationWithVarMeanRunner<Gemm> gemm;

  auto status = gemm.run(rankA, dstA, rankB, dstB, rankC, dstC, rankD, dstD,
                         rankVar, dstVar, rankMean, dstMean, alpha, beta, eps);

  assert(status == cutlass::Status::kSuccess);
}

void nvteGemmF16(int64_t rankA, void *dstA, bool transa, int64_t rankB,
                 void *dstB, bool transb, int64_t rankC, void *dstC,
                 int64_t rankD, void *dstD, int64_t activation, float alpha,
                 float beta, int32_t, int32_t) {
  using namespace mlir::hands_on_mlir::homnvgpu_kernel;
  GemmNVTERunner<half> gemm;

  checkCudaErrors(cudaStreamSynchronize(nullptr));

  auto status = gemm.run(rankA, dstA, transa, rankB, dstB, transb, rankC, dstC,
                         rankD, dstD, activation, alpha, beta);

  return;

  cublasHandle_t handle;
  checkCuBlasErrors(cublasCreate(&handle));

  auto a = static_cast<StridedMemRefType<half, 3> *>(dstA);
  auto b = static_cast<StridedMemRefType<half, 3> *>(dstB);
  auto c = static_cast<StridedMemRefType<half, 3> *>(dstC);
  auto d = allocHelper<half, 3>({c->sizes[0], c->sizes[1], c->sizes[2]},
                                nvgpuAllocer);
  auto desD = static_cast<StridedMemRefType<half, 3> *>(d.descriptor);
  auto trueD = static_cast<StridedMemRefType<half, 3> *>(dstD);

  cudaMemcpy(desD->data, c->data,
             sizeof(half) * c->sizes[0] * c->sizes[1] * c->sizes[2],
             cudaMemcpyDeviceToDevice);

  half al = alpha, be = beta;

  checkCudaErrors(cudaStreamSynchronize(nullptr));

  if (desD->sizes[2] != b->sizes[1]) {
    for (int i = 0; i < 3; i++) {
      std::cout << a->sizes[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 3; i++) {
      std::cout << b->sizes[i] << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 3; i++) {
      std::cout << c->sizes[i] << " ";
    }
    std::cout << std::endl;
    exit(0);
  }

  cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, b->sizes[1],
              a->sizes[0] * a->sizes[1], a->sizes[2], &al, b->data, b->sizes[2],
              a->data, a->sizes[2], &be, desD->data, desD->sizes[2]);

  checkCudaErrors(cudaStreamSynchronize(nullptr));

  // checkSame<half>(3, d.descriptor, 3, dstD);

  cudaMemcpy(trueD->data, desD->data,
             sizeof(half) * c->sizes[0] * c->sizes[1] * c->sizes[2],
             cudaMemcpyDeviceToDevice);

  cudaFree(desD->data);
  free(desD);

  cublasDestroy(handle);

  // assert(status == cutlass::Status::kSuccess);
}

void cutlassLayernormGemmF32(int64_t rankA, void *dstA, int64_t rankB,
                             void *dstB, int64_t rankC, void *dstC,
                             int64_t rankD, void *dstD, int64_t rankVar,
                             void *dstVar, int64_t rankMean, void *dstMean,
                             float alpha, float beta, int64_t activation) {}

void cutlassLayernormGemmF16(int64_t rankA, void *dstA, int64_t rankB,
                             void *dstB, int64_t rankC, void *dstC,
                             int64_t rankD, void *dstD, int64_t rankVar,
                             void *dstVar, int64_t rankMean, void *dstMean,
                             float alpha, float beta, int64_t activation) {
  using namespace mlir::hands_on_mlir::homnvgpu_kernel;

  using RowMajor = cutlass::layout::RowMajor;

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value,
      cutlass::half_t, cutlass::half_t>;

  using GemmMainloopFusion =
      typename cutlass::gemm::device::GemmLayernormMainloopFusion<
          cutlass::half_t, RowMajor, cutlass::half_t, RowMajor, cutlass::half_t,
          RowMajor, cutlass::half_t, RowMajor, cutlass::half_t,
          cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape,
          WarpShape, InstructionShape, EpilogueFunctorOp,
          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  LayernormGemmOperationRunner<GemmMainloopFusion> runner;

  auto status = runner.run(rankA, dstA, rankB, dstB, rankC, dstC, rankD, dstD,
                           rankVar, dstVar, rankMean, dstMean, alpha, beta);

  assert(status == Status::kSuccess);
}

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
  using namespace mlir::hands_on_mlir::homnvgpu_kernel;
  LayernormRunner<float> lnRunner;
  lnRunner.run(rankA, dstA, eps);
}

void nvteLayernormF16(int64_t rankA, void *dstA, float eps) {
  using namespace mlir::hands_on_mlir::homnvgpu_kernel;
  LayernormRunner<half> lnRunner;
  lnRunner.run(rankA, dstA, eps);
}

void nvteBertAttentionF32(int64_t rankA, void *dstA, int64_t rankSeqlen,
                          void *dstSeqlen, int64_t rankOut, void *dstOut,
                          float scale, int64_t headNum) {
  using namespace mlir::hands_on_mlir::homnvgpu_kernel;
  BertAttentionRunner<float> bertAttnRunner;
  bertAttnRunner.run(rankA, dstA, rankSeqlen, dstSeqlen, rankOut, dstOut, scale,
                     headNum);
}

void nvteBertAttentionF16(int64_t rankA, void *dstA, int64_t rankSeqlen,
                          void *dstSeqlen, int64_t rankOut, void *dstOut,
                          float scale, int64_t headNum) {
  using namespace mlir::hands_on_mlir::homnvgpu_kernel;
  BertAttentionRunner<half> bertAttnRunner;
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
