#include "ExecutionEngine/ExecutionEngine.h"
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/Utils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/Support/Error.h"
#include <cstdlib>
#include <cuda.h>
#include <iostream>

struct Res {
  C_UnrankedMemRefType a;
};

#define RowMajor(A, des, i, j, k)                                              \
  ((A)[(i) * (des).strides[0] + (j) * (des).strides[1] +                       \
       (k) * (des).strides[2]])

int main() {
  C_UnrankedMemRefType a;

  a.rank = 3;

  a.descriptor = malloc(sizeof(StridedMemRefType<float, 3>));
  auto des = static_cast<StridedMemRefType<float, 3> *>(a.descriptor);
  auto host_ptr = new float[3 * 200000];
  checkCudaErrors(cudaMalloc(&(des->data), sizeof(float) * 3 * 200000));
  std::cout << des->data << std::endl;
  des->basePtr = des->data;
  des->sizes[0] = 2;
  des->sizes[1] = 3;
  des->sizes[2] = 100000;
  des->strides[0] = 300000;
  des->strides[1] = 100000;
  des->strides[2] = 1;
  for (int i = 0; i < 600000; i++) {
    host_ptr[i] = 1;
  }
  cudaMemcpy(des->data, host_ptr, sizeof(float) * 3 * 200000,
             cudaMemcpyHostToDevice);

  Res b;
  mlir::hands_on_mlir::ExecutionEngine e("liblinear_nvgpu.so");

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

  auto c = DynamicMemRefType<float>(b.a);
  std::cout << c.rank << std::endl;
  cudaMemcpy(host_ptr, c.data,
             sizeof(float) * c.sizes[0] * c.sizes[1] * c.sizes[2],
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < c.sizes[0]; i++) {
    for (int j = 0; j < c.sizes[1]; j++) {
      for (int k = 0; k < c.sizes[2]; k++) {
        std::cout << RowMajor(host_ptr, c, i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  cudaFree(des->data);
  cudaFree(c.data);

  delete[] host_ptr;

  free(a.descriptor);
  free(b.a.descriptor);
}
