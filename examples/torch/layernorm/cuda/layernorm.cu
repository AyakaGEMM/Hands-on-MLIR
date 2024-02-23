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

#define RowMajor(A, des, i, j)                                                 \
  ((A)[(i) * (des).strides[0] + (j) * (des).strides[1]])

int main() {
  C_UnrankedMemRefType a;

  a.rank = 2;

  a.descriptor = malloc(sizeof(StridedMemRefType<float, 2>));
  auto des = static_cast<StridedMemRefType<float, 2> *>(a.descriptor);
  float host_ptr[] = {0.8823, 0.9150, 0.3829, 0.9593, 0.3904, 0.6009, 0.2566,
                      0.7936, 0.9408, 0.1332, 0.9346, 0.5936, 0.8694, 0.5677,
                      0.7411, 0.4294, 0.8854, 0.5739, 0.2666, 0.6274};
  checkCudaErrors(cudaMalloc(&(des->data), sizeof(host_ptr)));
  std::cout << des->data << std::endl;
  des->basePtr = des->data;
  des->sizes[0] = 2;
  des->sizes[1] = 10;
  des->strides[0] = 10;
  des->strides[1] = 1;
  cudaMemcpy(des->data, host_ptr, sizeof(host_ptr), cudaMemcpyHostToDevice);

  Res b;
  mlir::hands_on_mlir::ExecutionEngine e("liblayernorm_nvgpu.so");

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
  std::cout << c.rank << " " << c.sizes[0] << " " << c.sizes[1] << std::endl;
  cudaMemcpy(host_ptr, c.data, sizeof(float) * c.sizes[0] * c.sizes[1],
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < c.sizes[0]; i++) {
    for (int j = 0; j < c.sizes[1]; j++) {
      std::cout << RowMajor(host_ptr, c, i, j) << " ";
    }
    std::cout << std::endl;
  }

  cudaFree(des->data);
  cudaFree(c.data);

  free(a.descriptor);
  free(b.a.descriptor);
}
