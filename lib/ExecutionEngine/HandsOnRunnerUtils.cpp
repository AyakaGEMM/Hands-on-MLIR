#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
using namespace std;

inline auto convertToDynamicMemRefType(int64_t rank, void *dst) {
  UnrankedMemRefType<float> unrankType = {rank, dst};
  DynamicMemRefType<float> dyType(unrankType);
  return dyType;
}

extern "C" void print2DMatrixF32(int64_t rank, void *dst) {
  assert(rank == 2);
  auto dyType = convertToDynamicMemRefType(rank, dst);
  cout << "Unranked Memref ";
  printMemRefMetaData(std::cout, dyType);
  cout << endl;
  for (int i = 0; i < dyType.sizes[0]; i++) {
    for (int j = 0; j < dyType.sizes[1]; j++) {
      cout << dyType.data[dyType.sizes[1] * i + j] << " ";
    }
    cout << endl;
  }
}
extern "C" void fill2DRandomMatrixF32(int64_t rank, void *dst) {
  assert(rank == 2);
  srand(time(nullptr));
  auto dyType = convertToDynamicMemRefType(rank, dst);
  float genRandMax = 5;
  for (int i = 0; i < dyType.sizes[0]; i++) {
    for (int j = 0; j < dyType.sizes[1]; j++) {
      dyType.data[dyType.sizes[1] * i + j] =
          (float)rand() / ((float)RAND_MAX / genRandMax);
    }
  }
}
extern "C" void fill2DIncMatrixF32(int64_t rank, void *dst) {
  assert(rank == 2);
  auto dyType = convertToDynamicMemRefType(rank, dst);
  int ii = 0;
  for (int i = 0; i < dyType.sizes[0]; i++) {
    for (int j = 0; j < dyType.sizes[1]; j++) {
      dyType.data[dyType.sizes[1] * i + j] = ++ii;
    }
  }
}
extern "C" void validateF32WithRefMatmul(int64_t rankA, void *dstA,
                                         int64_t rankB, void *dstB,
                                         int64_t rankC1, void *dstC1,
                                         int64_t rankC2, void *dstC2) {
  auto A = convertToDynamicMemRefType(rankA, dstA);
  auto B = convertToDynamicMemRefType(rankB, dstB);
  auto C1 = convertToDynamicMemRefType(rankC1, dstC1);
  auto C2 = convertToDynamicMemRefType(rankC2, dstC2);

  assert(A.sizes[1] == B.sizes[0]);
  assert(A.sizes[0] == C1.sizes[0]);
  assert(B.sizes[1] == C1.sizes[1]);
  assert(C2.sizes[0] == C1.sizes[0]);
  assert(C2.sizes[1] == C1.sizes[1]);

  cout << "Pass input check. " << endl;

  auto M = A.sizes[0];
  auto N = B.sizes[1];
  auto K = B.sizes[0];
  for (int64_t k = 0; k < K; k++) {
    for (int64_t i = 0; i < M; i++) {
      for (int64_t j = 0; j < N; j++) {
        C1.data[i * N + j] += A.data[i * K + k] * B.data[j + k * N];
      }
    }
  }

  cout << "Finished Naive GEMM Cal. " << endl;

  constexpr double eps = 1e-6;

  for (int64_t i = 0; i < M; i++) {
    for (int64_t j = 0; j < N; j++) {
      double abs_err = fabs(C1.data[i * N + j] - C2.data[i * N + j]);
      double dot_length = M;
      double abs_val = fabs(C1.data[i * N + j]);
      double rel_err = abs_err / abs_val / dot_length;
      if (rel_err > eps) {
        cerr << "Test not pass. First failed element at [" << i << ", " << j
             << "]. Expect " << C1.data[i * N + j] << ", but get "
             << C2.data[i * N + j] << ". " << endl;
        exit(-1);
      }
    }
  }

  cout << "Test Passed! " << endl;
}