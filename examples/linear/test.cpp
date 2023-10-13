#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>

extern "C" CUnrankedMemRefType forward(int64_t rank, void *dst);

#define RowMajor(A, i, j, k)                                                   \
  ((A).data[(i) * (A).strides[0] + (j) * (A).strides[1] + (k) * (A).strides[2]])

inline auto convertToDynamicMemRefType(int64_t rank, void *dst) {
  UnrankedMemRefType<float> unrankType = {rank, dst};
  DynamicMemRefType<float> dyType(unrankType);
  return dyType;
}

template <class T> struct Result {
  Result(T &v) : _v(v) {}
  T &result() { return _v; }
  T &_v;
};

int main() {
  CUnrankedMemRefType a;

  a.rank = 3;

  a.descriptor = malloc(sizeof(StridedMemRefType<float, 3>));
  auto des = static_cast<StridedMemRefType<float, 3> *>(a.descriptor);
  des->data = new float[3 * 4000];
  des->basePtr = des->data;
  des->sizes[0] = 1;
  des->sizes[1] = 3;
  des->sizes[2] = 4000;
  des->strides[0] = 12000;
  des->strides[1] = 4000;
  des->strides[2] = 1;
  for (int i = 0; i < 12000; i++) {
    des->data[i] = 1;
  }
  auto b = forward(a.rank, a.descriptor);
  auto c = DynamicMemRefType<float>(b);
  for (int i = 0; i < c.sizes[0]; i++) {
    for (int j = 0; j < c.sizes[1]; j++) {
      for (int k = 0; k < c.sizes[2]; k++) {
        std::cout << RowMajor(c, i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  delete[] des->data;
  free(a.descriptor);
  delete[] c.data;
  free(b.descriptor);
}