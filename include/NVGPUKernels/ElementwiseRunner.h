#pragma once

#include "NVGPUKernels/OperationRunner.h"
#include "NVGPUKernels/Utils.h"
#include "thrust/device_ptr.h"
#include "thrust/iterator/zip_iterator.h"
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

template <typename ElementType_>
struct Add : public std::unary_function<ElementType_, ElementType_> {
  using ElementType = ElementType_;
  __host__ __device__ constexpr ElementType
  operator()(const ElementType &a, const ElementType &b) const {
    return a + b;
  }
};

template <typename ElementType_>
struct Sub : public std::unary_function<ElementType_, ElementType_> {
  using ElementType = ElementType_;
  __host__ __device__ constexpr ElementType
  operator()(const ElementType &a, const ElementType &b) const {
    return a - b;
  }
};

template <typename ElementType_>
struct Mul : public std::unary_function<ElementType_, ElementType_> {
  using ElementType = ElementType_;
  __host__ __device__ constexpr ElementType
  operator()(const ElementType &a, const ElementType &b) const {
    return a * b;
  }
};

template <typename ElementType_>
struct Div : public std::unary_function<ElementType_, ElementType_> {
  using ElementType = ElementType_;
  __host__ __device__ constexpr ElementType
  operator()(const ElementType &a, const ElementType &b) const {
    return a / b;
  }
};

template <typename ElementwiseOp>
class ElementwiseRunner : public mlir::hands_on_mlir::OperationRunner {

public:
  template <typename T> struct doNothing : public std::unary_function<T, T> {
    __host__ __device__ constexpr T operator()(const T &x) const { return x; }
  };

  template <typename T>
  struct getRealVal : public std::unary_function<int64_t, T> {
  private:
    thrust::device_ptr<T> underlayingPtr_;

    int64_t rank_;

    int64_t *myShape_;
    int64_t *dstStride_;
    int64_t *myStride_;

  public:
    getRealVal(thrust::device_ptr<T> underlayingPtr, int64_t rank,
               const int64_t *dstStride, const int64_t *myShape,
               const int64_t *myStride)
        : underlayingPtr_(underlayingPtr) {

      auto workspace = getDummyPointer<int64_t>(rank * 3);
      rank_ = rank;

      dstStride_ = workspace.get();
      myStride_ = workspace.get() + rank;
      myShape_ = workspace.get() + rank * 2;

      checkCudaErrors(cudaMemcpyAsync(dstStride_, dstStride,
                                      sizeof(int64_t) * rank,
                                      cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpyAsync(
          myStride_, myStride, sizeof(int64_t) * rank, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpyAsync(myShape_, myShape, sizeof(int64_t) * rank,
                                      cudaMemcpyHostToDevice));
    }

    __host__ __device__ constexpr T operator()(const int64_t &x) const {
      int64_t finalCoord = 0;
      int64_t x_ = x;

      for (int64_t i = 0; i < rank_; i++) {
        auto coord = x_ / dstStride_[i];
        finalCoord +=
            (coord < myShape_[i] ? coord : myShape_[i] - 1) * myStride_[i];
        x_ %= dstStride_[i];
      }

      return *(underlayingPtr_ + finalCoord);
    }
  };

public:
  // To-do: I use thrust here only for fast development. Could protentially be
  // fused into a single kernel or at least use multi stream to improve the
  // performance.
  Status run(int rankA, void *desA, int rankB, void *desB, int rankOut,
             void *desOut) {
    auto A = convertToDynamicMemRefType<typename ElementwiseOp::ElementType>(
        rankA, desA);
    auto B = convertToDynamicMemRefType<typename ElementwiseOp::ElementType>(
        rankB, desB);
    auto Out = convertToDynamicMemRefType<typename ElementwiseOp::ElementType>(
        rankOut, desOut);

    auto inTotlaSize = std::accumulate(A.sizes, A.sizes + A.rank, 1,
                                       std::multiplies<int64_t>());

    assert(A.rank == B.rank);
    assert(A.rank == Out.rank);
    for (int i = 0; i < A.rank; i++) {
      assert(A.sizes[i] == B.sizes[i] ||
             B.sizes[i] == 1); // Only can broadcast B.
      assert(A.sizes[i] == Out.sizes[i]);
    }

    auto APtr = thrust::device_pointer_cast(A.data);
    auto BPtr = thrust::device_pointer_cast(B.data);
    auto OutPtr = thrust::device_pointer_cast(Out.data);

    auto BwithBroadCast = thrust::make_transform_iterator(
        thrust::make_counting_iterator<int64_t>(0),
        getRealVal<typename ElementwiseOp::ElementType>(BPtr, A.rank, A.strides,
                                                        B.sizes, B.strides));

    thrust::transform(APtr, APtr + inTotlaSize, BwithBroadCast, OutPtr,
                      ElementwiseOp());

    return Status::kSuccess;
  }
};

} // namespace homnvgpu_kernel
} // namespace hands_on_mlir
} // namespace mlir
