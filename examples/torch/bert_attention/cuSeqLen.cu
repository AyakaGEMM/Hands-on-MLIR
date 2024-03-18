#include "ExecutionEngine/ExecutionEngine.h"
#include "ExecutionEngine/HandsOnNVGPURunnerUtils.h"
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/Utils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <iostream>

#define RowMajor(A, des, i) ((A)[(i) * (des).strides[0]])

int main() {

  constexpr int64_t bs = 16;
  constexpr int64_t seq_len = 64;

  auto a = allocHelper<int64_t, 2, int64_t>({bs, seq_len}, nvgpuAllocer);

  auto des = static_cast<StridedMemRefType<int64_t, 2> *>(a.descriptor);

  auto host_ptr = new int64_t[seq_len * bs];
  checkCudaErrors(cudaMalloc(&(des->data), sizeof(int64_t) * bs * seq_len));

  int32_t data[] = {64, 12, 31, 32, 33, 34, 35, 36};
  for (int i = 0; i < bs; i++) {
    for (int j = 0; j < seq_len; j++) {
      host_ptr[i * seq_len + j] = j < data[i % 8];
      std::cout << host_ptr[i * seq_len + j] << " ";
    }
    std::cout << std::endl;
  }
  cudaMemcpy(des->data, host_ptr, sizeof(int64_t) * seq_len * bs,
             cudaMemcpyHostToDevice);

  UnrankedMemRefType<int32_t> b;
  mlir::hands_on_mlir::ExecutionEngine e("libcuseqlen_nvgpu.so");

  auto res = e.invoke("forward", a.rank, a.descriptor,
                      mlir::hands_on_mlir::ExecutionEngine::result(b));
  if (res) {
    llvm::handleAllErrors(std::move(res));
  }

  res = e.invoke("forward", a.rank, a.descriptor,
                 mlir::hands_on_mlir::ExecutionEngine::result(b));
  if (res) {
    llvm::handleAllErrors(std::move(res));
  }

  auto new_host = new int32_t[bs * seq_len];

  auto c = DynamicMemRefType<int32_t>(b);
  std::cout << c.rank << std::endl;
  cudaMemcpy(new_host, c.data, sizeof(int32_t) * c.sizes[0],
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < c.sizes[0]; i++) {
    std::cout << RowMajor(new_host, c, i) << " ";
  }
  std::cout << std::endl;

  cudaFree(des->data);
  cudaFree(c.data);

  delete[] host_ptr;
  delete[] new_host;

  free(a.descriptor);
  free(b.descriptor);
}
