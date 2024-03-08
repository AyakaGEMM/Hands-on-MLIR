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
#include <iostream>
#include <vector>

#define RowMajor(A, des, i, j, k)                                              \
  ((A)[(i) * (des).strides[0] + (j) * (des).strides[1] +                       \
       (k) * (des).strides[2]])

int main() {
  constexpr int64_t bs = 2;
  constexpr int64_t seq_len = 64;
  constexpr int64_t output_size = 30522;
  constexpr int64_t real_len = 10;
  auto input_ids =
      allocHelper<int64_t, 2, int64_t>({bs, seq_len}, nvgpuAllocer);
  auto mask = allocHelper<int64_t, 2, int64_t>({bs, seq_len}, nvgpuAllocer);
  auto type_id = allocHelper<int64_t, 2, int64_t>({bs, seq_len}, nvgpuAllocer);

  auto input_ids_des =
      static_cast<StridedMemRefType<int64_t, 2> *>(input_ids.descriptor);
  auto mask_des = static_cast<StridedMemRefType<int64_t, 2> *>(mask.descriptor);
  auto type_id_des =
      static_cast<StridedMemRefType<int64_t, 2> *>(type_id.descriptor);

  std::vector<int64_t> input_ids_data(seq_len * bs);

  std::ifstream in;
  in.open("0.txt");
  int64_t a;
  size_t ii = 0;
  while (in >> a) {
    assert(ii < input_ids_data.size());
    input_ids_data[ii++] = a;
  }
  in.close();

  checkCudaErrors(cudaMemcpy(input_ids_des->data, input_ids_data.data(),
                             sizeof(int64_t) * input_ids_data.size(),
                             cudaMemcpyHostToDevice));

  in.open("1.txt");
  ii = 0;
  while (in >> a) {
    assert(ii < input_ids_data.size());
    input_ids_data[ii++] = a;
  }
  in.close();

  checkCudaErrors(cudaMemcpy(mask_des->data, input_ids_data.data(),
                             sizeof(int64_t) * input_ids_data.size(),
                             cudaMemcpyHostToDevice));

  in.open("2.txt");
  ii = 0;
  while (in >> a) {
    assert(ii < input_ids_data.size());
    input_ids_data[ii++] = a;
  }

  checkCudaErrors(cudaMemcpy(type_id_des->data, input_ids_data.data(),
                             sizeof(int64_t) * input_ids_data.size(),
                             cudaMemcpyHostToDevice));

  UnrankedMemRefType<half> b;
  mlir::hands_on_mlir::ExecutionEngine e("libbert_nvgpu.so");

  // Warm up
  auto res =
      e.invoke("forward", input_ids.rank, input_ids.descriptor, mask.rank,
               mask.descriptor, type_id.rank, type_id.descriptor,
               mlir::hands_on_mlir::ExecutionEngine::result(b));
  if (res) {
    llvm::handleAllErrors(std::move(res));
  }

  for (int i = 0; i < 10; i++) {
    res = e.invoke("forward", input_ids.rank, input_ids.descriptor, mask.rank,
                   mask.descriptor, type_id.rank, type_id.descriptor,
                   mlir::hands_on_mlir::ExecutionEngine::result(b));
    if (res) {
      llvm::handleAllErrors(std::move(res));
    }
  }

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start));

  for (int i = 0; i < 1000; i++) {
    res = e.invoke("forward", input_ids.rank, input_ids.descriptor, mask.rank,
                   mask.descriptor, type_id.rank, type_id.descriptor,
                   mlir::hands_on_mlir::ExecutionEngine::result(b));
    if (res) {
      llvm::handleAllErrors(std::move(res));
    }
  }

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  float msecTotal = 0;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  std::cout << "E2E latency: " << msecTotal / 1000 / 1000 << "s" << std::endl;

  auto c = DynamicMemRefType<half>(b);
  std::cout << c.rank << std::endl;

  std::vector<half> thing;
  in.close();
  in.open("3.txt");
  float bb;
  while (in >> bb) {
    thing.emplace_back(bb);
  }

  half *data = new half[bs * seq_len * output_size];

  checkCudaErrors(cudaMemcpy(data, c.data,
                             sizeof(half) * bs * seq_len * output_size,
                             cudaMemcpyDeviceToHost));
  for (int i = 0; i < c.sizes[0]; i++) {
    for (int j = 0; j < real_len; j++) {
      for (int k = 0; k < c.sizes[2]; k++) {
        if (std::abs(float(RowMajor(data, c, i, j, k) -
                           RowMajor(thing, c, i, j, k))) > 1e-2) {
          std::cout << "Not ok" << std::endl;
        }
      }
    }
  }

  cudaFree(c.data);

  free(input_ids.descriptor);
  free(mask.descriptor);
  free(b.descriptor);
}
