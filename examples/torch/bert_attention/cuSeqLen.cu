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

struct Res {
  UnrankedMemRefType<int32_t> a;
};

#define RowMajor(A, des, i) ((A)[(i) * (des).strides[0]])

int main() {

  constexpr int64_t bs = 1;
  constexpr int64_t seq_len = 64;

  auto a = allocHelper<int32_t, 2, int32_t>({bs, seq_len}, nvgpuAllocer);

  auto des = static_cast<StridedMemRefType<int32_t, 2> *>(a.descriptor);

  auto host_ptr = new int32_t[seq_len * bs];
  checkCudaErrors(cudaMalloc(&(des->data), sizeof(int32_t) * bs * seq_len));

  int32_t data[] = {64};
  for (int i = 0; i < bs; i++) {
    for (int j = 0; j < seq_len; j++) {
      host_ptr[i * seq_len + j] = j < data[i];
      std::cout << host_ptr[i * seq_len + j] << " ";
    }
    std::cout << std::endl;
  }
  cudaMemcpy(des->data, host_ptr, sizeof(int32_t) * seq_len * bs,
             cudaMemcpyHostToDevice);

  Res b;
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

  auto c = DynamicMemRefType<int32_t>(b.a);
  std::cout << c.rank << std::endl;
  cudaMemcpy(host_ptr, c.data, sizeof(int32_t) * c.sizes[0],
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < c.sizes[0]; i++) {
    std::cout << RowMajor(host_ptr, c, i) << " ";
  }
  std::cout << std::endl;

  cudaFree(des->data);
  cudaFree(c.data);

  delete[] host_ptr;

  free(a.descriptor);
  free(b.a.descriptor);
}
