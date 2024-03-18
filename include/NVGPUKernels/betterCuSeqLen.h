#pragma once

#include "NVGPUKernels/OperationRunner.h"
#include "NVGPUKernels/Utils.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/discard_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/reduce.h"
#include <cstdint>
#include <cuda_runtime_api.h>

#include <functional>
#include <numeric>
#include <nvToolsExt.h>
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

  struct MakeKey : public std::unary_function<int64_t, int64_t> {

    int64_t row_length_;

    MakeKey(int64_t row_length) : row_length_(row_length) {}

    __host__ __device__ constexpr int64_t operator()(const int64_t &x) const {
      return (x / row_length_) & 1;
    }
  };

  template <typename Arg, typename Result>
  struct Cast : public std::unary_function<Arg, Result> {

    __host__ __device__ constexpr Result operator()(const Arg &x) const {
      return Result(x);
    }
  };

public:
  // To-do: I use thrust here only for fast development. Could protentially be
  // fused into a single kernel or at least use multi stream to improve the
  // performance.
  // To-do(short-term plan): use reduce by key
  Status run(int rankIn, void *desIn, int rankOut, void *desOut) {
    auto In = convertToDynamicMemRefType<InputElementType>(rankIn, desIn);
    auto Out = convertToDynamicMemRefType<int32_t>(rankOut, desOut);

    auto inTotlaSize = std::accumulate(In.sizes, In.sizes + rankIn, 1,
                                       std::multiplies<int64_t>());

    assert(In.sizes[0] + 1 == Out.sizes[0]);
    assert(In.rank == 2);
    assert(Out.rank == 1);
    checkCudaErrors(cudaMemset(
        Out.data, 0, sizeof(int32_t))); // Use memset to avoid malloc by thrust

    auto key = thrust::make_transform_iterator(
        thrust::make_counting_iterator<int64_t>(0), MakeKey(In.sizes[1]));
    auto inPtr = thrust::device_pointer_cast(In.data);
    auto in = thrust::make_transform_iterator(
        inPtr, Cast<InputElementType, int32_t>());
    auto outPtr = thrust::device_pointer_cast(Out.data);

    thrust::reduce_by_key(key, key + inTotlaSize, in,
                          thrust::make_discard_iterator(), outPtr + 1);

    thrust::inclusive_scan(outPtr + 1, outPtr + In.sizes[0] + 1, outPtr + 1);

    checkCudaErrors(cudaGetLastError());

    return Status::kSuccess;
  }
};

} // namespace homnvgpu_kernel
} // namespace hands_on_mlir
} // namespace mlir
