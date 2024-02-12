#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

template <class T = float>
inline auto convertToDynamicMemRefType(int64_t rank, void *dst) {
  UnrankedMemRefType<T> unrankType = {rank, dst};
  DynamicMemRefType<T> dyType(unrankType);
  return dyType;
}

template <class T, int rank>
void *createStridedMemRef(StridedMemRefType<T, rank> *src) {
  auto desVoid = malloc(sizeof(StridedMemRefType<T, rank>));
  auto des = static_cast<StridedMemRefType<T, rank> *>(desVoid);
  int64_t elementNum = 1;
  for (int64_t i = 0; i < rank; i++) {
    des->sizes[i] = src->sizes[i];
    des->strides[i] = src->strides[i];
    elementNum *= src->sizes[i];
  }
  des->data = new T[elementNum];
  des->offset = 0;
  des->basePtr = des->data;
  return static_cast<void *>(des);
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

extern "C" void deallocF32(int64_t rank, void *dst) {
  auto memRef = convertToDynamicMemRefType(rank, dst);
  delete[] memRef.data;
}

#define RowMajor(A, i, j, k)                                                   \
  ((A).data[(i) * (A).strides[0] + (j) * (A).strides[1] + (k) * (A).strides[2]])

extern "C" C_UnrankedMemRefType allocF32(int32_t elementNum) {
  auto returnMemRef = C_UnrankedMemRefType();
  returnMemRef.rank = 1;
  returnMemRef.descriptor = malloc(sizeof(
      StridedMemRefType<float, 1>)); // MLIR will delete this ptr for us. Also,
                                     // we have to use malloc here since mlir
                                     // will use free to delete it.
  auto des =
      static_cast<StridedMemRefType<float, 1> *>(returnMemRef.descriptor);
  des->sizes[0] = elementNum;
  des->strides[0] = 1;
  des->data = new float[elementNum];
  des->basePtr = des->data;
  des->offset = 0;
  return returnMemRef;
}

extern "C" C_UnrankedMemRefType alloc3DMemRefF32(int32_t a, int32_t b,
                                                 int32_t c) {
  auto returnMemRef = C_UnrankedMemRefType();
  returnMemRef.rank = 3;
  returnMemRef.descriptor = malloc(sizeof(StridedMemRefType<float, 3>));
  auto des =
      static_cast<StridedMemRefType<float, 3> *>(returnMemRef.descriptor);
  des->data = new float[a * b * c];
  des->basePtr = des->data;
  des->offset = 0;
  des->sizes[0] = a;
  des->sizes[1] = b;
  des->sizes[2] = c;
  des->strides[0] = b * c;
  des->strides[1] = c;
  des->strides[2] = 1;
  return returnMemRef;
}

extern "C" C_UnrankedMemRefType allocByMemRefF32(int64_t rank, void *dst) {
  auto returnMemRef = C_UnrankedMemRefType();
  returnMemRef.rank = rank;
  assert(rank <= 5 && rank > 0);

  switch (rank) {
  case 1:
    returnMemRef.descriptor =
        createStridedMemRef(static_cast<StridedMemRefType<float, 1> *>(dst));
    break;
  case 2:
    returnMemRef.descriptor =
        createStridedMemRef(static_cast<StridedMemRefType<float, 2> *>(dst));
    break;
  case 3:
    returnMemRef.descriptor =
        createStridedMemRef(static_cast<StridedMemRefType<float, 3> *>(dst));
    break;
  case 4:
    returnMemRef.descriptor =
        createStridedMemRef(static_cast<StridedMemRefType<float, 4> *>(dst));
    break;
  case 5:
    returnMemRef.descriptor =
        createStridedMemRef(static_cast<StridedMemRefType<float, 5> *>(dst));
    break;
  default:
    break;
  }

  return returnMemRef;
}

extern "C" C_UnrankedMemRefType allocConstantF32(int32_t idx) {
  auto haha = C_UnrankedMemRefType();
  // So stupid...
  ifstream file(filesystem::path(__FILE__).parent_path().string() +
                string("/../../examples/torch/linear/") + to_string(idx) +
                ".txt");
  vector<int> v;
  int a;
  string line;
  getline(file, line);
  stringstream ss(line);
  while (ss >> a) {
    v.push_back(a);
  }
  haha.rank = v.size();
  haha.descriptor = malloc(
      sizeof(StridedMemRefType<float, 3>)); // MLIR will delete this ptr for us.
  auto des = static_cast<StridedMemRefType<float, 3> *>(haha.descriptor);
  int32_t stride = 1;
  for (size_t i = 0; i < v.size(); i++) {
    des->sizes[i] = v[i];
    des->strides[2 - i] = stride;
    stride *= v[2 - i];
  }
  des->data = new float[stride];
  for (int i = 0; i < stride; i++) {
    file >> des->data[i];
  }
  des->basePtr = des->data;
  des->offset = 0;
  return haha;
}

extern "C" void matmulAddF32(int64_t rankA, void *dstA, int64_t rankB,
                             void *dstB, int64_t rankC, void *dstC,
                             int64_t rankD, void *dstD) {
  auto A = convertToDynamicMemRefType(rankA, dstA);
  auto B = convertToDynamicMemRefType(rankB, dstB);
  auto C = convertToDynamicMemRefType(rankC, dstC);
  auto D = convertToDynamicMemRefType(rankD, dstD);

  std::cout << rankB << std::endl;

  assert(rankA == 3);
  assert(rankB == 3);
  assert(rankC == 3);
  assert(rankD == 3);

  assert(B.sizes[0] == 1);
  assert(C.sizes[0] == 1);

  assert(A.sizes[0] == B.sizes[0] || B.sizes[0] == 1);
  assert(A.sizes[0] == C.sizes[0] || C.sizes[0] == 1);
  assert(A.sizes[0] == D.sizes[0]);

  assert(A.sizes[2] == B.sizes[1]);
  assert(A.sizes[1] == C.sizes[1] || C.sizes[1] == 1);
  assert(B.sizes[2] == C.sizes[2]);

  assert(A.sizes[1] == D.sizes[1]);
  assert(B.sizes[2] == D.sizes[2]);

  const auto M = D.sizes[1];
  const auto N = D.sizes[2];
  const auto K = A.sizes[2];

  for (int i = 0; i < D.sizes[0]; i++) {
    for (int j = 0; j < D.sizes[1]; j++) {
      for (int k = 0; k < D.sizes[2]; k++) {
        RowMajor(D, i, j, k) =
            C.data[(C.sizes[0] == 1 ? 0 : i) * C.strides[0] +
                   (C.sizes[1] == 1 ? 0 : j) * C.strides[1] + k * C.strides[2]];
      }
    }
  }

  for (int b = 0; b < D.sizes[0]; b++) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
          RowMajor(D, b, i, j) += RowMajor(A, 0, i, k) * RowMajor(B, 0, k, j);
        }
      }
    }
  }
}
