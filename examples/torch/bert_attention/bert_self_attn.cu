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
#include <functional>
#include <iostream>
#include <numeric>

struct Res {
  UnrankedMemRefType<int32_t> a;
};

#define RowMajor(A, des, i, j, k)                                              \
  ((A)[(i) * (des).strides[0] + (j) * (des).strides[1] +                       \
       (k) * (des).strides[0]])

int main() {
  constexpr int64_t seq_len = 64;
  auto hidden_state =
      allocHelper<half, 3, half>({1, seq_len, 128}, nvgpuAllocer);
  auto mask = allocHelper<int32_t, 2>({1, seq_len}, nvgpuAllocer);

  auto hidden_des =
      static_cast<StridedMemRefType<half, 3> *>(hidden_state.descriptor);
  auto mask_des =
      static_cast<StridedMemRefType<int32_t, 2> *>(hidden_state.descriptor);

  half hidden_data[] = {
      0.8823, 0.9150, 0.3829, 0.9593, 0.3904, 0.6009, 0.2566, 0.7936, 0.9408,
      0.1332, 0.9346, 0.5936, 0.8694, 0.5677, 0.7411, 0.4294, 0.8854, 0.5739,
      0.2666, 0.6274, 0.2696, 0.4414, 0.2969, 0.8317, 0.1053, 0.2695, 0.3588,
      0.1994, 0.5472, 0.0062, 0.9516, 0.0753, 0.8860, 0.5832, 0.3376, 0.8090,
      0.5779, 0.9040, 0.5547, 0.3423, 0.6343, 0.3644, 0.7104, 0.9464, 0.7890,
      0.2814, 0.7886, 0.5895, 0.7539, 0.1952, 0.0050, 0.3068, 0.1165, 0.9103,
      0.6440, 0.7071, 0.6581, 0.4913, 0.8913, 0.1447, 0.5315, 0.1587, 0.6542,
      0.3278, 0.6532, 0.3958, 0.9147, 0.2036, 0.2018, 0.2018, 0.9497, 0.6666,
      0.9811, 0.0874, 0.0041, 0.1088, 0.1637, 0.7025, 0.6790, 0.9155, 0.2418,
      0.1591, 0.7653, 0.2979, 0.8035, 0.3813, 0.7860, 0.1115, 0.2477, 0.6524,
      0.6057, 0.3725, 0.7980, 0.8399, 0.1374, 0.2331, 0.9578, 0.3313, 0.3227,
      0.0162, 0.2137, 0.6249, 0.4340, 0.1371, 0.5117, 0.1585, 0.0758, 0.2247,
      0.0624, 0.1816, 0.9998, 0.5944, 0.6541, 0.0337, 0.1716, 0.3336, 0.5782,
      0.0600, 0.2846, 0.2007, 0.5014, 0.3139, 0.4654, 0.1612, 0.1568, 0.2083,
      0.3289, 0.1054, 0.9192, 0.4008, 0.9302, 0.6558, 0.0766, 0.8460, 0.3624,
      0.3083, 0.0850, 0.0029, 0.6431, 0.3908, 0.6947, 0.0897, 0.8712, 0.1330,
      0.4137, 0.6044, 0.7581, 0.9037, 0.9555, 0.1035, 0.6258, 0.2849, 0.4452,
      0.1258, 0.9554, 0.1330, 0.7672, 0.6757, 0.6625, 0.2297, 0.9545, 0.6099,
      0.5643, 0.0594, 0.7099, 0.4250, 0.2709, 0.9295, 0.6115, 0.2234, 0.2469,
      0.4761, 0.7792, 0.3722, 0.2147, 0.3288, 0.1265, 0.6783, 0.8870, 0.0293,
      0.6161, 0.7583, 0.5907, 0.3219, 0.7610, 0.7628, 0.6870, 0.4121, 0.3676,
      0.5535, 0.4117, 0.3510, 0.8196, 0.9297, 0.4505, 0.3881, 0.5073, 0.4701,
      0.6202, 0.6401, 0.0459, 0.3155, 0.9211, 0.6948, 0.4751, 0.1985, 0.1941,
      0.0521, 0.3370, 0.6689, 0.8188, 0.7308, 0.0580, 0.1993, 0.4211, 0.9837,
      0.5723, 0.3705, 0.7069, 0.3096, 0.1764, 0.8649, 0.2726, 0.3998, 0.0026,
      0.8346, 0.8788, 0.6822, 0.1514, 0.0065, 0.0939, 0.8729, 0.7401, 0.9208,
      0.7619, 0.6265, 0.4951, 0.1197, 0.0716, 0.0323, 0.7047, 0.2545, 0.3994,
      0.2122, 0.4089, 0.1481, 0.1733, 0.6659, 0.3514, 0.8087, 0.3396, 0.1332,
      0.4118, 0.2576, 0.3470, 0.0240};

  checkCudaErrors(cudaMemcpy(hidden_des->data, hidden_data, sizeof(hidden_data),
                             cudaMemcpyHostToDevice));

  int32_t mask_data[seq_len];
  memset(mask_data, 0, sizeof(mask_data));
  checkCudaErrors(cudaMemcpy(mask_des->data, mask_data, sizeof(mask_data),
                             cudaMemcpyHostToDevice));

  UnrankedMemRefType<half> b;
  mlir::hands_on_mlir::ExecutionEngine e("libbert_self_attn_nvgpu.so");

  auto res = e.invoke("forward", hidden_state.rank, hidden_state.descriptor,
                      mask.rank, mask.descriptor,
                      mlir::hands_on_mlir::ExecutionEngine::result(b));
  if (res) {
    llvm::handleAllErrors(std::move(res));
  }

  res = e.invoke("forward", hidden_state.rank, hidden_state.descriptor,
                 mask.rank, mask.descriptor,
                 mlir::hands_on_mlir::ExecutionEngine::result(b));
  if (res) {
    llvm::handleAllErrors(std::move(res));
  }

  auto c = DynamicMemRefType<half>(b);
  std::cout << c.rank << std::endl;
  cudaMemcpy(hidden_data, c.data,
             sizeof(half) * std::accumulate(c.sizes, c.sizes + c.rank, 1,
                                            std::multiplies<>()),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < c.sizes[0]; i++) {
    for (int j = 0; j < c.sizes[1]; j++) {
      for (int k = 0; k < c.sizes[2]; k++) {
        std::cout << float(RowMajor(hidden_data, c, i, j, k)) << " ";
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
