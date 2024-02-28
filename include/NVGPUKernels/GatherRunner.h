#pragma once

#include "NVGPUKernels/OperationRunner.h"
#include "NVGPUKernels/Utils.h"
#include "thrust/device_ptr.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/zip_iterator.h"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <thrust/gather.h>

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

template <typename ElementType>
class GatherRunner : public mlir::hands_on_mlir::OperationRunner {
public:
  template <typename T = int64_t> struct a : public std::unary_function<T, T> {

    thrust::device_ptr<T> indices;
    T row_length;

    __host__ __device__ constexpr T operator()(const T &x) const {
      return x % row_length + indices[x / row_length];
    }

    a(thrust::device_ptr<T> indices_, T row_length_)
        : indices(indices), row_length(row_length) {}
  };

public:
  // To-do: I use thrust here only for fast development. Could protentially be
  // fused into a single kernel or at least use multi stream to improve the
  // performance.
  Status run(int rankIndices, void *desIndices, int rankValue, void *desValue,
             int rankOut, void *desOut) {
    auto indices = convertToDynamicMemRefType<int64_t>(rankIndices, desIndices);
    auto value = convertToDynamicMemRefType<ElementType>(rankValue, desValue);
    auto out = convertToDynamicMemRefType<ElementType>(rankOut, desOut);

    assert(value.rank == 3);
    assert(value.sizes[0] == 1);
    assert(indices.rank == out.rank);
    for (auto i = 0; i < indices.rank; i++) {
      assert(indices.sizes[i] == out.sizes[i]);
    }

    auto indices_thrust_ptr = thrust::device_pointer_cast(indices.data);
    auto value_thrust_ptr = thrust::device_pointer_cast(value.data);
    auto out_thrust_ptr = thrust::device_pointer_cast(out.data);

    auto total_size = std::accumulate(
        indices.sizes, indices.sizes + indices.rank, 1, std::multiplies<>());

    auto map_iter =
        thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                        a(indices_thrust_ptr, value.sizes[2]));

    thrust::gather(map_iter, map_iter + total_size, value_thrust_ptr,
                   out_thrust_ptr);

    return Status::kSuccess;
  }
};

} // namespace homnvgpu_kernel
} // namespace hands_on_mlir
} // namespace mlir
