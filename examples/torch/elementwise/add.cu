#include "ExecutionEngine/ExecutionEngine.h"
#include "ExecutionEngine/HandsOnNVGPURunnerUtils.h"
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/Utils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/Support/Error.h"
#include <cstdlib>
#include <cuda.h>
#include <iostream>

#define RowMajor(A, des, i, j, k)                                              \
  ((A)[(i) * (des).strides[0] + (j) * (des).strides[1] +                       \
       (k) * (des).strides[2]])

int main() {

  auto a = allocHelper<float, 3>({3, 3, 3}, nvgpuAllocer);
  auto b = allocHelper<float, 3>({3, 1, 3}, nvgpuAllocer);

  auto Ades = static_cast<StridedMemRefType<float, 3> *>(a.descriptor);
  float host_ptr_a[] = {0.8823, 0.9150, 0.3829, 0.9593, 0.3904, 0.6009, 0.2566,
                        0.7936, 0.9408, 0.1332, 0.9346, 0.5936, 0.8694, 0.5677,
                        0.7411, 0.4294, 0.8854, 0.5739, 0.2666, 0.6274, 0.2696,
                        0.4414, 0.2969, 0.8317, 0.1053, 0.2695, 0.3588};
  float host_ptr_b[] = {0.1994, 0.5472, 0.0062, 0.9516, 0.0753,
                        0.8860, 0.5832, 0.3376, 0.809};
  cudaMemcpy(Ades->data, host_ptr_a, sizeof(host_ptr_a),
             cudaMemcpyHostToDevice);

  auto Bdes = static_cast<StridedMemRefType<float, 3> *>(b.descriptor);
  cudaMemcpy(Bdes->data, host_ptr_b, sizeof(host_ptr_b),
             cudaMemcpyHostToDevice);

  UnrankedMemRefType<float> cc;
  mlir::hands_on_mlir::ExecutionEngine e("libadd_nvgpu.so");

  auto res = e.invoke("forward", a.rank, a.descriptor, b.rank, b.descriptor,
                      mlir::hands_on_mlir::ExecutionEngine::result(cc));
  if (res) {
    llvm::handleAllErrors(std::move(res));
  }

  res = e.invoke("forward", a.rank, a.descriptor, b.rank, b.descriptor,
                 mlir::hands_on_mlir::ExecutionEngine::result(b));
  if (res) {
    llvm::handleAllErrors(std::move(res));
  }

  auto c = DynamicMemRefType<float>(cc);
  std::cout << c.rank << " " << c.sizes[0] << " " << c.sizes[1] << " "
            << c.sizes[2] << std::endl;
  cudaMemcpy(host_ptr_a, c.data,
             sizeof(float) * c.sizes[0] * c.sizes[1] * c.sizes[2],
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < c.sizes[0]; i++) {
    for (int j = 0; j < c.sizes[1]; j++) {
      for (int k = 0; k < c.sizes[2]; k++) {
        std::cout << RowMajor(host_ptr_a, c, i, j, k) << " ";
      }
    }
    std::cout << std::endl;
  }

  cudaFree(Ades->data);
  cudaFree(Bdes->data);
  cudaFree(c.data);

  free(a.descriptor);
  free(b.descriptor);
  free(cc.descriptor);
}
