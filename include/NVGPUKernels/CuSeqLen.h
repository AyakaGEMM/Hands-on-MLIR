#pragma once

#include "NVGPUKernels/OperationRunner.h"
#include "NVGPUKernels/Utils.h"
#include <cstdint>
#include <cuda_runtime_api.h>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/util_allocator.cuh>
#include <functional>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_input_output_iterator.h>
#include <thrust/transform.h>

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu_kernel {

template <typename InputElementType>
class CuSeqLenRunner : public mlir::hands_on_mlir::OperationRunner {

public:
  template <typename Arg, typename Result>
  struct flip : public std::unary_function<Arg, Result> {
    __host__ __device__ constexpr Result operator()(const Arg &x) const {
      return Arg(1) - x;
    }
  };

public:
  // To-do: I use thrust here only for fast development. Could protentially be
  // fused into a single kernel or at least use multi stream to improve the
  // performance.
  Status run(int rankIn, void *desIn, int rankOut, void *desOut) {
    auto In = convertToDynamicMemRefType<InputElementType>(rankIn, desIn);
    auto Out = convertToDynamicMemRefType<int32_t>(rankOut, desOut);

    auto inTotlaSize = std::accumulate(In.sizes, In.sizes + rankIn, 1,
                                       std::multiplies<int64_t>());

    assert(In.sizes[0] + 1 == Out.sizes[0]);
    assert(In.rank == 2);
    assert(Out.rank == 1);

    for (int64_t i = 0; i < In.sizes[0]; i++) {
      auto inPtr = thrust::device_pointer_cast(In.data + i * In.strides[0]);
      auto outPtr =
          thrust::device_pointer_cast(Out.data + (i + 1) * Out.strides[0]);

      auto inIter =
          thrust::make_transform_iterator(inPtr, flip<int32_t, int32_t>());

      *outPtr = thrust::reduce(inIter, inIter + In.strides[0]);
    }

    auto outPtr = thrust::device_pointer_cast(Out.data);

    *outPtr = 0;
    thrust::inclusive_scan(outPtr + 1, outPtr + In.sizes[0] + 1, outPtr + 1);

    return Status::kSuccess;
  }
};

} // namespace homnvgpu_kernel
} // namespace hands_on_mlir
} // namespace mlir
