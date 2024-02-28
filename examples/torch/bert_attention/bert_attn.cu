#include "ExecutionEngine/ExecutionEngine.h"
#include "ExecutionEngine/HandsOnNVGPURunnerUtils.h"
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/Utils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/Support/Error.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#define RowMajor(A, des, i, j, k)                                              \
  ((A)[(i) * (des).strides[0] + (j) * (des).strides[1] +                       \
       (k) * (des).strides[2]])

int main() {
  constexpr int64_t seq_len = 64;
  constexpr int64_t hidden_size = 768;
  auto hidden_state =
      allocHelper<half, 3, half>({1, seq_len, hidden_size}, nvgpuAllocer);
  auto mask = allocHelper<int32_t, 2>({1, seq_len}, nvgpuAllocer);

  auto hidden_des =
      static_cast<StridedMemRefType<half, 3> *>(hidden_state.descriptor);
  auto mask_des = static_cast<StridedMemRefType<int32_t, 2> *>(mask.descriptor);

  std::vector<half> hidden_data(hidden_size * seq_len);

  std::ifstream in;
  in.open("0.txt");
  float a;
  size_t ii = 0;
  while (in >> a) {
    assert(ii < hidden_data.size());
    hidden_data[ii++] = a;
  }

  checkCudaErrors(cudaMemcpy(hidden_des->data, hidden_data.data(),
                             sizeof(half) * hidden_data.size(),
                             cudaMemcpyHostToDevice));

  int32_t mask_data[seq_len];
  for (auto &i : mask_data) {
    i = 1;
  }
  checkCudaErrors(cudaMemcpy(mask_des->data, mask_data, sizeof(mask_data),
                             cudaMemcpyHostToDevice));

  UnrankedMemRefType<half> b;
  mlir::hands_on_mlir::ExecutionEngine e("libbert_attn_nvgpu.so");

  auto res = e.invoke("forward", hidden_state.rank, hidden_state.descriptor,
                      mask.rank, mask.descriptor,
                      mlir::hands_on_mlir::ExecutionEngine::result(b));
  if (res) {
    llvm::handleAllErrors(std::move(res));
  }

  // res = e.invoke("forward", hidden_state.rank, hidden_state.descriptor,
  //                mask.rank, mask.descriptor,
  //                mlir::hands_on_mlir::ExecutionEngine::result(b));
  // if (res) {
  //   llvm::handleAllErrors(std::move(res));
  // }

  auto c = DynamicMemRefType<half>(b);
  std::cout << c.rank << std::endl;
  assert(std::accumulate(c.sizes, c.sizes + c.rank, 1, std::multiplies<>()) ==
         hidden_data.size());

  std::vector<half> thing;
  in.close();
  in.open("1.txt");
  while (in >> a) {
    thing.emplace_back(a);
  }
  checkCudaErrors(cudaMemcpy(hidden_data.data(), c.data,
                             sizeof(half) * hidden_data.size(),
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < c.sizes[0]; i++) {
    for (int j = 0; j < c.sizes[1]; j++) {
      for (int k = 0; k < c.sizes[2]; k++) {
        std::cout << float(RowMajor(hidden_data, c, i, j, k) -
                           RowMajor(thing, c, i, j, k))
                  << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  cudaFree(hidden_des->data);
  cudaFree(c.data);

  free(hidden_state.descriptor);
  free(mask.descriptor);
  free(b.descriptor);
}
