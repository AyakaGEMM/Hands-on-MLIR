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
        a[i * (n * k) + j * k + ii] = float(rand()) / ((float)RAND_MAX / 5);
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

  int m = 8, n = 8, k = 8;

  C_UnrankedMemRefType a, b, c, mean, var;
  a.rank = b.rank = c.rank = 3;
  mean.rank = var.rank = 1;
  a.descriptor = malloc(sizeof(StridedMemRefType<int16_t, 3>));
  b.descriptor = malloc(sizeof(StridedMemRefType<int16_t, 3>));
  c.descriptor = malloc(sizeof(StridedMemRefType<int16_t, 3>));
  mean.descriptor = malloc(sizeof(StridedMemRefType<int16_t, 1>));
  var.descriptor = malloc(sizeof(StridedMemRefType<int16_t, 1>));
  _Float16 *host_ptr = new _Float16[2 * m * n * k];

  auto des = static_cast<StridedMemRefType<_Float16, 3> *>(a.descriptor);
  checkCudaErrors(cudaMalloc(&(des->data), sizeof(_Float16) * 2 * m * k));
  fillRand(host_ptr, 2, m, k);
  print3D(host_ptr, 2, m, k);
  checkCudaErrors(cudaMemcpy(des->data, host_ptr, sizeof(int16_t) * 2 * m * k,
                             cudaMemcpyHostToDevice));
  des->sizes[0] = 2;
  des->sizes[1] = m;
  des->sizes[2] = k;

  des = static_cast<StridedMemRefType<_Float16, 3> *>(b.descriptor);
  checkCudaErrors(cudaMalloc(&(des->data), sizeof(_Float16) * 1 * k * n));
  fillRand(host_ptr, 1, k, n);
  print3D(host_ptr, 1, k, n);
  checkCudaErrors(cudaMemcpy(des->data, host_ptr, sizeof(int16_t) * 1 * k * n,
                             cudaMemcpyHostToDevice));
  des->sizes[0] = 1;
  des->sizes[1] = k;
  des->sizes[2] = n;

  des = static_cast<StridedMemRefType<_Float16, 3> *>(c.descriptor);
  checkCudaErrors(cudaMalloc(&(des->data), sizeof(_Float16) * 2 * m * n));
  fillRand(host_ptr, 2, m, n);
  checkCudaErrors(cudaMemcpy(des->data, host_ptr, sizeof(int16_t) * 2 * m * n,
                             cudaMemcpyHostToDevice));
  des->sizes[0] = 2;
  des->sizes[1] = m;
  des->sizes[2] = n;

  auto desMean = static_cast<StridedMemRefType<_Float16, 1> *>(mean.descriptor);
  checkCudaErrors(cudaMalloc(&(desMean->data), sizeof(_Float16) * 2 * n));
  fillRand(host_ptr, 1, 1, 2 * n);
  checkCudaErrors(cudaMemcpy(desMean->data, host_ptr, sizeof(int16_t) * 2 * n,
                             cudaMemcpyHostToDevice));
  desMean->sizes[0] = 2 * n;
  print1D(host_ptr, 2 * n);

  auto desVar = static_cast<StridedMemRefType<_Float16, 1> *>(var.descriptor);
  checkCudaErrors(cudaMalloc(&(desVar->data), sizeof(_Float16) * 2 * n));
  plusOne(host_ptr, host_ptr, 1, 1, 2 * n);
  checkCudaErrors(cudaMemcpy(desVar->data, host_ptr, sizeof(int16_t) * 2 * n,
                             cudaMemcpyHostToDevice));
  print1D(host_ptr, 2 * n);
  desVar->sizes[0] = 2 * n;

  cutlassGemmWithVarMeanF16(a.rank, a.descriptor, b.rank, b.descriptor, c.rank,
                            c.descriptor, c.rank, c.descriptor, var.rank,
                            var.descriptor, mean.rank, mean.descriptor, 1, 0, 0,
                            1e-6);

  checkCudaErrors(cudaMemcpy(host_ptr, des->data, sizeof(_Float16) * 2 * m * n,
                             cudaMemcpyDeviceToHost));
  print3D(host_ptr, 2, m, n);

  checkCudaErrors(cudaMemcpy(host_ptr, desMean->data, sizeof(_Float16) * 2 * n,
                             cudaMemcpyDeviceToHost));
  print1D(host_ptr, 2 * n);

  checkCudaErrors(cudaMemcpy(host_ptr, desVar->data, sizeof(_Float16) * 2 * n,
                             cudaMemcpyDeviceToHost));
  print1D(host_ptr, 2 * n);
}
