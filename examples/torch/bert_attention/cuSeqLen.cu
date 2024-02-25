#include "ExecutionEngine/ExecutionEngine.h"
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
  C_UnrankedMemRefType a;

  a.rank = 2;
  a.descriptor = malloc(sizeof(StridedMemRefType<int32_t, 2>));
  auto des = static_cast<StridedMemRefType<int32_t, 2> *>(a.descriptor);
  auto host_ptr = new int32_t[16 * 3];
  checkCudaErrors(cudaMalloc(&(des->data), sizeof(int32_t) * 16 * 3));
  std::cout << des->data << std::endl;
  des->basePtr = des->data;
  des->sizes[0] = 3;
  des->sizes[1] = 16;
  des->strides[0] = 16;
  des->strides[1] = 1;
  int32_t data[] = {3, 16, 12};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 16; j++) {
      host_ptr[i * 16 + j] = 1 - (j < data[i]);
      std::cout << host_ptr[i * 16 + j] << " ";
    }
    std::cout << std::endl;
  }
  cudaMemcpy(des->data, host_ptr, sizeof(int32_t) * 16 * 3,
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
