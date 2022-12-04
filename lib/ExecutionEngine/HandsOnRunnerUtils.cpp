#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include <cstdlib>
#include <iostream>

extern "C" void fillRandomMatrix(float *dst, size_t M, size_t N) {}
extern "C" void validateWithRefImpl(float *A, float *B, float *C1, float *C2,
                                    size_t M, size_t N, size_t K) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      for (size_t k = 0; k < K; k++) {
        C1[i * N + j] += A[i * K + k] * B[j + k * N];
      }
    }
  }
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      if (C1[i * N + j] != C2[i * N + j]) {
        exit(-1);
      }
    }
  }
}