#include "ExecutionEngine/ExecutionEngine.h"
#include "ExecutionEngine/HandsOnNVGPURunnerUtils.h"
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/Utils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/Support/Error.h"
#include <cstdlib>
#include <cuda.h>
#include <iostream>

struct Res {
  UnrankedMemRefType<half> a;
};

#define RowMajor(A, des, i, j, k)                                              \
  ((A)[(i) * (des).strides[0] + (j) * (des).strides[1] +                       \
       (k) * (des).strides[2]])

int main(int argc, char *argv[]) {

  auto m = std::atoi(argv[1]), n = std::atoi(argv[2]), k = std::atoi(argv[3]);

  auto a = allocHelper<half, 3>({1, m, k}, nvgpuAllocer);

  Res b;

  std::string filename = "libhom_linear_" + std::to_string(m) + "_" +
                         std::to_string(n) + "_" + std::to_string(k) + ".so";

  mlir::hands_on_mlir::ExecutionEngine e(filename);
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  auto res = e.invoke("forward", a.rank, a.descriptor,
                      mlir::hands_on_mlir::ExecutionEngine::result(b));
  if (res) {
    llvm::handleAllErrors(std::move(res));
  }

  checkCudaErrors(cudaEventRecord(start));

  for (int i = 0; i < 1000; i++) {
    res = e.invoke("forward", a.rank, a.descriptor,
                   mlir::hands_on_mlir::ExecutionEngine::result(b));
    if (res) {
      llvm::handleAllErrors(std::move(res));
    }
  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  std::cout << "E2E latency: " << msecTotal / 1000.0 << "ms" << std::endl;

  free(a.descriptor);
  free(b.a.descriptor);
}
