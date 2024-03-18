#include "ExecutionEngine/ExecutionEngine.h"
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/Utils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "transformer_engine/gemm.h"
#include "transformer_engine/transformer_engine.h"
#include "llvm/Support/Error.h"
#include <cstdlib>
#include <cuda.h>
#include <iostream>

int main() {
  using namespace transformer_engine;

  unsigned long long m = 2048, n = 3072, k = 768;
  half *a, *b, *c;
  checkCudaErrors(cudaMalloc(&a, sizeof(half) * m * k));
  checkCudaErrors(cudaMalloc(&b, sizeof(half) * n * k));
  checkCudaErrors(cudaMalloc(&c, sizeof(half) * m * n));

  TensorWrapper a_tensor(a, {m, k}, NVTEWrapperDTypeMap<half>::kType);
  TensorWrapper b_tensor(b, {n, k}, NVTEWrapperDTypeMap<half>::kType);
  TensorWrapper c_tensor(c, {m, n}, NVTEWrapperDTypeMap<half>::kType);

  auto workspace_buffer = getDummyPointer(4 * 1024 * 1024);
  auto pre_gelu_buffer = nullptr;

  TensorWrapper workspace(workspace_buffer.get(), {4 * 1024 * 1024},
                          NVTEWrapperDTypeMap<char>::kType);
  TensorWrapper bias(nullptr, std::vector<size_t>{0},
                     NVTEWrapperDTypeMap<half>::kType);
  TensorWrapper pre_gelu(pre_gelu_buffer, std::vector<size_t>{0},
                         NVTEWrapperDTypeMap<half>::kType);

  auto mpCount = getMulitProcessorCount();
  nvte_cublas_gemm(b_tensor.data(), a_tensor.data(), c_tensor.data(),
                   bias.data(), pre_gelu.data(), true, false, false,
                   workspace.data(), false, false, mpCount, nullptr);

  cudaEvent_t s, t;
  checkCudaErrors(cudaEventCreate(&s));
  checkCudaErrors(cudaEventCreate(&t));

  checkCudaErrors(cudaEventRecord(s));

  for (int i = 0; i < 10000; i++) {
    nvte_cublas_gemm(b_tensor.data(), a_tensor.data(), c_tensor.data(),
                     bias.data(), pre_gelu.data(), true, false, false,
                     workspace.data(), false, false, mpCount, nullptr);
  }

  cudaEventRecord(t);
  checkCudaErrors(cudaEventSynchronize(t));
  float msecTotal = 0;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, s, t));

  std::cout << "E2E latency: " << msecTotal / 1000.0 / 10000.0 << "s"
            << std::endl;
  std::cout << "GFlops: "
            << (m * n * k * 2.0) * 1e-9 / (msecTotal / 10000.0 / 1000.0)
            << std::endl;
}
