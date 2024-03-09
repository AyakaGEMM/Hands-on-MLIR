#include "ExecutionEngine/HandsOnNVGPURunnerUtils.h"
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/Utils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <iostream>

void fillRand(_Float16 *a, int64_t m, int64_t n, int64_t k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int ii = 0; ii < k; ii++) {
        a[i * (n * k) + j * k + ii] = float(rand()) / ((float)RAND_MAX / 1);
      }
    }
  }
}

void plusOne(_Float16 *a, _Float16 *b, int64_t m, int64_t n, int64_t k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int ii = 0; ii < k; ii++) {
        a[i * (n * k) + j * k + ii] =
            b[i * (n * k) + j * k + ii] * b[i * (n * k) + j * k + ii] + 1;
      }
    }
  }
}

void print3D(_Float16 *a, int64_t m, int64_t n, int64_t k) {

  std::cout << "=====================\n";
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int ii = 0; ii < k; ii++) {
        std::cout << float(a[i * (n * k) + j * k + ii]) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << "=====================\n";
}

void print1D(_Float16 *a, int64_t m) {
  std::cout << "=====================\n";
  for (int i = 0; i < m; i++) {
    std::cout << float(a[i]) << " ";
  }
  std::cout << "\n=====================\n";
}

int main() {

  int bs = 2;

  int m = 2, n = 8, k = 32;
  int scale = 2;

  auto a = allocHelper<half, 3, half>({bs, m, k}, nvgpuAllocer);
  auto b = allocHelper<half, 3, half>({1, k, n}, nvgpuAllocer);
  auto c = allocHelper<half, 3, half>({bs, m, n}, nvgpuAllocer);
  auto d = allocHelper<half, 3, half>({1, n, n * scale}, nvgpuAllocer);
  auto e = allocHelper<half, 3, half>({bs, m, n * scale}, nvgpuAllocer);

  auto mean = allocHelper<half, 1, half>({bs * m}, nvgpuAllocer);
  auto var = allocHelper<half, 1, half>({bs * m}, nvgpuAllocer);

  _Float16 *host_ptr = new _Float16[bs * m * n * k * scale];

  auto des = static_cast<StridedMemRefType<_Float16, 3> *>(a.descriptor);
  fillRand(host_ptr, des->sizes[0], des->sizes[1], des->sizes[2]);
  print3D(host_ptr, bs, m, k);
  checkCudaErrors(cudaMemcpy(des->data, host_ptr, sizeof(int16_t) * bs * m * k,
                             cudaMemcpyHostToDevice));

  des = static_cast<StridedMemRefType<_Float16, 3> *>(b.descriptor);
  fillRand(host_ptr, 1, k, n);
  print3D(host_ptr, 1, k, n);
  checkCudaErrors(cudaMemcpy(des->data, host_ptr, sizeof(int16_t) * 1 * k * n,
                             cudaMemcpyHostToDevice));

  des = static_cast<StridedMemRefType<_Float16, 3> *>(d.descriptor);
  fillRand(host_ptr, 1, n, n * scale);
  print3D(host_ptr, 1, n, n * scale);
  checkCudaErrors(cudaMemcpy(des->data, host_ptr,
                             sizeof(int16_t) * 1 * n * n * scale,
                             cudaMemcpyHostToDevice));

  auto desMean = static_cast<StridedMemRefType<_Float16, 1> *>(mean.descriptor);
  auto desVar = static_cast<StridedMemRefType<_Float16, 1> *>(var.descriptor);

  cutlassGemmWithVarMeanF16(a.rank, a.descriptor, b.rank, b.descriptor, c.rank,
                            c.descriptor, c.rank, c.descriptor, var.rank,
                            var.descriptor, mean.rank, mean.descriptor, 1, 0, 0,
                            1e-6);

  cutlassLayernormGemmF16(c.rank, c.descriptor, d.rank, d.descriptor, e.rank,
                          e.descriptor, e.rank, e.descriptor, var.rank,
                          var.descriptor, mean.rank, mean.descriptor, 1, 0, 0);

  des = static_cast<StridedMemRefType<_Float16, 3> *>(c.descriptor);

  checkCudaErrors(cudaMemcpy(host_ptr, des->data, sizeof(_Float16) * bs * m * n,
                             cudaMemcpyDeviceToHost));
  print3D(host_ptr, bs, m, n);

  checkCudaErrors(cudaMemcpy(host_ptr, desMean->data, sizeof(_Float16) * bs * m,
                             cudaMemcpyDeviceToHost));
  print1D(host_ptr, bs * m);

  checkCudaErrors(cudaMemcpy(host_ptr, desVar->data, sizeof(_Float16) * bs * m,
                             cudaMemcpyDeviceToHost));
  print1D(host_ptr, bs * m);

  des = static_cast<StridedMemRefType<_Float16, 3> *>(e.descriptor);

  checkCudaErrors(cudaMemcpy(host_ptr, des->data,
                             sizeof(_Float16) * bs * m * n * scale,
                             cudaMemcpyDeviceToHost));
  print3D(host_ptr, bs, m, n * scale);
}
