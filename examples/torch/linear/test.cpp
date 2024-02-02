#include "ExecutionEngine/ExecutionEngine.h"
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>

struct Res {
  CUnrankedMemRefType a;
};

#define RowMajor(A, i, j, k)                                                   \
  ((A).data[(i) * (A).strides[0] + (j) * (A).strides[1] + (k) * (A).strides[2]])

int main() {
  CUnrankedMemRefType a;

  a.rank = 3;

  a.descriptor = malloc(sizeof(StridedMemRefType<float, 3>));
  auto des = static_cast<StridedMemRefType<float, 3> *>(a.descriptor);
  des->data = new float[3 * 100];
  des->basePtr = des->data;
  des->sizes[0] = 1;
  des->sizes[1] = 3;
  des->sizes[2] = 100;
  des->strides[0] = 300;
  des->strides[1] = 100;
  des->strides[2] = 1;
  for (int i = 0; i < 300; i++) {
    des->data[i] = 1;
  }

  Res b;
  mlir::hands_on_mlir::ExecutionEngine e("liblinear.so");

  auto res = e.invoke("forward", a.rank, a.descriptor,
                      mlir::hands_on_mlir::ExecutionEngine::result(b));
  if (res) {
    llvm::handleAllErrors(std::move(res));
  }
  auto c = DynamicMemRefType<float>(b.a);
  std::cout << c.rank << std::endl;
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
  delete[] c.data;

  free(a.descriptor);
  free(b.a.descriptor);
}
