#include "ExecutionEngine/ExecutionEngine.h"
#include "ExecutionEngine/HandsOnNVGPURunnerUtils.h"
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/Support/Error.h"
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define RowMajor(A, des, i, j, k)                                              \
  ((A)[(i) * (des).strides[0] + (j) * (des).strides[1] +                       \
       (k) * (des).strides[2]])

int main() {

  int64_t m = 64 * 8, n = 768, k = 768;

  auto a = allocHelper<half, 3, half>({1, m, k}, nvgpuAllocer);
  auto b = allocHelper<half, 3, half>({1, k, n}, nvgpuAllocer);
  auto c = allocHelper<half, 3, half>({1, m, n}, nvgpuAllocer);
  auto d = allocHelper<half, 3, half>({1, m, n}, nvgpuAllocer);

  auto a_host = new half[m * k];
  auto b_host = new half[n * k];
  auto c_host = new half[m * n];

  for (int i = 0; i < m * k; i++) {
    a_host[i] = 0.1;
  }

  for (int i = 0; i < n * k; i++) {
    b_host[i] = 0.2;
  }

  for (int i = 0; i < m * n; i++) {
    c_host[i] = 0.1;
  }

  auto desA = static_cast<StridedMemRefType<half, 3> *>(a.descriptor);
  auto desb = static_cast<StridedMemRefType<half, 3> *>(b.descriptor);
  auto desd = static_cast<StridedMemRefType<half, 3> *>(d.descriptor);

  cudaMemcpy(desA->data, a_host, sizeof(half) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(desb->data, b_host, sizeof(half) * n * k, cudaMemcpyHostToDevice);

  cutlassGemmF16(a.rank, a.descriptor, false, b.rank, b.descriptor, false,
                 c.rank, c.descriptor, d.rank, d.descriptor, 0, 1, 0, 219, 1);

  auto err = cudaStreamSynchronize(nullptr);
  if (err != cudaSuccess) {
    std::cout << cudaGetErrorString(err) << std::endl;
    exit(-1);
  }

  cudaMemcpy(c_host, desd->data, sizeof(half) * m * n, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < m; j++) {
      for (int kk = 0; kk < n; kk++) {
        if (std::abs(float(c_host[i]) - k * 0.02) > 1e-1) {
          std::cout << float(c_host[i]) << std::endl;
          std::cout << float(c_host[i]) - k * 0.01 << std::endl;
          std::cout << "Not ok" << std::endl;
        }
      }
    }
  }
}
