#pragma once

#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/OperationRunner.h"
#include "NVGPUKernels/Utils.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/library/descriptions.h"
#include "transformer_engine/gemm.h"
#include "transformer_engine/transformer_engine.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu_kernel {

/// Structure describing the tiled structure of a GEMM-like computation
struct TileDescription {

  /// Describes the shape of a threadblock (in elements)
  cutlass::gemm::GemmCoord threadblock_shape;

  /// Describes the number of pipeline stages in the threadblock-scoped mainloop
  int threadblock_stages;

  /// Number of warps in each logical dimension
  cutlass::gemm::GemmCoord warp_count;

  /// Core math instruction
  MathInstructionDescription math_instruction;

  /// Minimum compute capability (e.g. 70, 75) of a device eligible to run the
  /// operation.
  int minimum_compute_capability;

  /// Minimum compute capability (e.g. 70, 75) of a device eligible to run the
  /// operation.
  int maximum_compute_capability;

  /// Describes the shape of a cluster (in blocks)
  cutlass::gemm::GemmCoord cluster_shape;

  //
  // Methods
  //

  TileDescription(
      cutlass::gemm::GemmCoord threadblock_shape = cutlass::gemm::GemmCoord(),
      int threadblock_stages = 0,
      cutlass::gemm::GemmCoord warp_count = cutlass::gemm::GemmCoord(),
      MathInstructionDescription math_instruction =
          MathInstructionDescription(),
      int minimum_compute_capability = 0, int maximum_compute_capability = 0,
      cutlass::gemm::GemmCoord cluster_shape = cutlass::gemm::GemmCoord(1, 1,
                                                                        1))
      : threadblock_shape(threadblock_shape),
        threadblock_stages(threadblock_stages), warp_count(warp_count),
        math_instruction(math_instruction),
        minimum_compute_capability(minimum_compute_capability),
        maximum_compute_capability(maximum_compute_capability),
        cluster_shape(cluster_shape) {}

  // Equality operator
  inline bool operator==(TileDescription const &rhs) const {
    return ((threadblock_shape == rhs.threadblock_shape) &&
            (threadblock_stages == rhs.threadblock_stages) &&
            (warp_count == rhs.warp_count) &&
            (math_instruction == rhs.math_instruction) &&
            (minimum_compute_capability == rhs.minimum_compute_capability) &&
            (maximum_compute_capability == rhs.maximum_compute_capability));
  }

  // Inequality operator
  inline bool operator!=(TileDescription const &rhs) const {
    return !(*this == rhs);
  }
};

template <typename Element, typename Layout>
cutlass::library::TensorDescription make_TensorDescription(int alignment = 1) {
  cutlass::library::TensorDescription desc;

  desc.element = NumericTypeMap<Element>::kId;
  desc.layout = LayoutMap<Layout>::kId;
  desc.alignment = alignment;
  desc.log_extent_range =
      int(sizeof(typename Layout::TensorCoord::Index) - 1) * 8;
  desc.log_stride_range = int(sizeof(typename Layout::Stride::Index) - 1) * 8;

  return desc;
}

struct GemmDescription {
  const char *name;
  cutlass::library::Provider provider;
  cutlass::library::OperationKind kind;
  cutlass::library::GemmKind gemm_kind;
  TileDescription tile_description;

  /// Describes the A operand
  cutlass::library::TensorDescription A;

  /// Describes the B operand
  cutlass::library::TensorDescription B;

  /// Describes the source matrix
  cutlass::library::TensorDescription C;

  /// Describes the destination matrix
  cutlass::library::TensorDescription D;
};

class GemmOperationRunnerBase {

protected:
  GemmDescription description_;

public:
  virtual Status run(int64_t rankA, void *dstA, int64_t rankB, void *dstB,
                     int64_t rankC, void *dstC, int64_t rankD, void *dstD,
                     float alpha, float beta, int32_t splitKFactor) = 0;

  virtual bool contains(const char *str);

  virtual bool isF16() = 0;

  virtual const GemmDescription &getGemmDescription() { return description_; }

  virtual ~GemmOperationRunnerBase() = 0;
};

template <typename Operator_>
class GemmOperationRunner : public GemmOperationRunnerBase {
public:
  using Operator = Operator_;
  using ElementA = typename Operator::ElementA;
  using LayoutA = typename Operator::LayoutA;
  using ElementB = typename Operator::ElementB;
  using LayoutB = typename Operator::LayoutB;
  using ElementC = typename Operator::ElementC;
  using LayoutC = typename Operator::LayoutC;
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  // assuming all tensors use same type for StrideIndex
  using StrideIndex = typename Operator::LayoutA::Index;
  using ElementAccumulator = typename Operator::ElementAccumulator;
  using ElementCompute = typename Operator::EpilogueOutputOp::ElementCompute;

  using OperatorArguments = typename Operator::Arguments;

protected:
  void *host_workspace_;
  void *device_workspace_;
  size_t device_workspace_size_;

public:
  /// Constructor
  GemmOperationRunner(char const *name = "unknown_gemm") {
    description_.name = name;
    description_.provider = cutlass::library::Provider::kCUTLASS;
    description_.kind = cutlass::library::OperationKind::kGemm;
    description_.gemm_kind = cutlass::library::GemmKind::kGemm;

    description_.tile_description.threadblock_shape = cutlass::make_Coord(
        Operator::ThreadblockShape::kM, Operator::ThreadblockShape::kN,
        Operator::ThreadblockShape::kK);

    description_.tile_description.threadblock_stages = Operator::kStages;

    description_.tile_description.warp_count =
        cutlass::make_Coord(Operator::GemmKernel::WarpCount::kM,
                            Operator::GemmKernel::WarpCount::kN,
                            Operator::GemmKernel::WarpCount::kK);

    description_.tile_description.math_instruction.instruction_shape =
        cutlass::make_Coord(Operator::InstructionShape::kM,
                            Operator::InstructionShape::kN,
                            Operator::InstructionShape::kK);

    host_workspace_ = device_workspace_ = nullptr;
    device_workspace_size_ = 0;

    description_.A =
        make_TensorDescription<ElementA, LayoutA>(Operator::kAlignmentA);
    description_.B =
        make_TensorDescription<ElementB, LayoutB>(Operator::kAlignmentB);
    description_.C =
        make_TensorDescription<ElementC, LayoutC>(Operator::kAlignmentC);
    description_.D =
        make_TensorDescription<ElementD, LayoutD>(Operator::kAlignmentC);
  }

  virtual ~GemmOperationRunner() {
    if (host_workspace_ != nullptr) {
      auto op = static_cast<Operator *>(host_workspace_);
      delete op;
    }

    if (device_workspace_size_ > 0) {
      cudaFree(device_workspace_);
    }
  }

  Status initialize_and_update(const OperatorArguments &args) {
    if (host_workspace_ == nullptr) {
      host_workspace_ = new Operator;
    }
    Operator *op = static_cast<Operator *>(host_workspace_);
    auto status = op->can_implement(args);

    if (status != Status::kSuccess) {
      return status;
    }

    auto required_workspace = op->get_workspace_size(args);

    if (required_workspace > device_workspace_size_) {
      if (device_workspace_size_ != 0) {
        cudaFree(device_workspace_);
      }
      auto error = cudaMalloc(&device_workspace_, required_workspace);
      assert(error == cudaSuccess);
      device_workspace_size_ = required_workspace;
    }

    return op->initialize(args, device_workspace_);
  }

  static Status construct_arguments(int64_t rankA, void *dstA, int64_t rankB,
                                    void *dstB, int64_t rankC, void *dstC,
                                    int64_t rankD, void *dstD, float alpha,
                                    float beta, int32_t splitKFactor,
                                    OperatorArguments &args) {
    auto A = convertToDynamicMemRefType<ElementA>(rankA, dstA);
    auto B = convertToDynamicMemRefType<ElementB>(rankB, dstB);
    auto C = convertToDynamicMemRefType<ElementC>(rankC, dstC);
    auto D = convertToDynamicMemRefType<ElementD>(rankD, dstD);

    assert(A.rank == 3);
    assert(B.rank == 3);
    assert(C.rank == 3);
    assert(D.rank == 3);

    auto DimCompatibale = [](int64_t a, int64_t b) {
      return a == b || a == 1 || b == 1;
    };

    assert(A.sizes[1] == C.sizes[1]);
    assert(B.sizes[2] == C.sizes[2]);
    assert(A.sizes[2] == B.sizes[1]);

    auto M = std::max(A.sizes[1], C.sizes[1]);
    auto N = std::max(B.sizes[2], C.sizes[2]);
    auto K = A.sizes[2];
    auto BatchSize = A.sizes[0];

    assert(DimCompatibale(BatchSize, B.sizes[0]));
    assert(BatchSize == C.sizes[0]);
    assert(BatchSize == D.sizes[0]);
    assert(C.sizes[0] == D.sizes[0]);
    assert(C.sizes[1] == D.sizes[1]);
    assert(C.sizes[2] == D.sizes[2]);

    args =
        OperatorArguments(cutlass::gemm::GemmUniversalMode::kGemm,
                          {int(M * BatchSize), int(N), int(K)}, splitKFactor,
                          {ElementCompute(alpha), ElementCompute(beta)}, A.data,
                          B.data, C.data, D.data, BatchSize * M * K, N * K,
                          BatchSize * M * N, BatchSize * M * N, K, N, N, N);

    return Status::kSuccess;
  }

  Status run(int64_t rankA, void *dstA, int64_t rankB, void *dstB,
             int64_t rankC, void *dstC, int64_t rankD, void *dstD, float alpha,
             float beta, int32_t splitKFactor) override {

    OperatorArguments args;

    auto status =
        construct_arguments(rankA, dstA, rankB, dstB, rankC, dstC, rankD, dstD,
                            alpha, beta, splitKFactor, args);
    if (status != Status::kSuccess) {
      return status;
    }

    status = initialize_and_update(args);
    if (status != Status::kSuccess) {
      return status;
    }

    auto op = static_cast<Operator *>(host_workspace_);

    return op->run();
  }

  virtual bool isF16() override {
    return (description_.A.element == NumericTypeID::kF16) &&
           (description_.B.element == NumericTypeID::kF16) &&
           (description_.C.element == NumericTypeID::kF16) &&
           (description_.D.element == NumericTypeID::kF16);
  }
};

template <typename ElementType> class GemmNVTERunner : public OperationRunner {

  using TensorWrapper = transformer_engine::TensorWrapper;

  std::tuple<TensorWrapper, TensorWrapper, TensorWrapper, TensorWrapper,
             TensorWrapper, TensorWrapper, bool>
  construct_tensors(int64_t rankA, void *dstA, bool transa, int64_t rankB,
                    void *dstB, bool transb, int64_t rankC, void *dstC,
                    int64_t rankD, void *dstD, int64_t activation, float alpha,
                    float beta) {

    auto A = convertToDynamicMemRefType(rankA, dstA);
    auto B = convertToDynamicMemRefType(rankB, dstB);
    auto C = convertToDynamicMemRefType(rankC, dstC);
    auto D = convertToDynamicMemRefType(rankD, dstD);

    assert(A.rank == 3);
    assert(B.rank == 3);
    assert(C.rank == 3);
    assert(D.rank == 3);
    assert(transa == false);
    assert(transb == true);

    auto DimCompatibale = [](int64_t a, int64_t b) {
      return a == b || a == 1 || b == 1;
    };

    assert(A.sizes[1] == C.sizes[1]);
    assert(B.sizes[1] == C.sizes[2]);
    assert(A.sizes[2] == B.sizes[2]);

    size_t M = std::max(A.sizes[1], C.sizes[1]);
    size_t N = std::max(B.sizes[1], C.sizes[2]);
    size_t K = A.sizes[2];
    auto BatchSize = A.sizes[0];

    assert(B.sizes[0] == 1);
    assert(BatchSize == C.sizes[0]);
    assert(BatchSize == D.sizes[0]);
    assert(C.sizes[0] == D.sizes[0]);
    assert(C.sizes[1] == D.sizes[1]);
    assert(C.sizes[2] == D.sizes[2]);

    assert(alpha == 1);

    std::vector<size_t> a_shape = {M * BatchSize, K};
    std::vector<size_t> b_shape = {N, K};
    std::vector<size_t> c_shape = {M, N};

    TensorWrapper a(A.data, a_shape, NVTEWrapperDTypeMap<ElementType>::kType);
    TensorWrapper b(B.data, b_shape, NVTEWrapperDTypeMap<ElementType>::kType);
    TensorWrapper c(D.data, c_shape, NVTEWrapperDTypeMap<ElementType>::kType);

    bool accumulate = false;

    if (beta != 0) {
      assert(beta == 1);
      assert(C.data == D.data);
      accumulate = true;
    }

    // Magic number here. Transformer engine uses 4MB for all architecture
    // except for Hopper.
    auto workspace_buffer = getDummyPointer(4 * 1024 * 1024);
    auto pre_gelu_buffer =
        activation == 1 ? getDummyPointer<ElementType>(M * N) : nullptr;

    TensorWrapper workspace(workspace_buffer.get(), {4 * 1024 * 1024},
                            NVTEWrapperDTypeMap<char>::kType);
    TensorWrapper bias(nullptr, std::vector<size_t>{0},
                       NVTEWrapperDTypeMap<ElementType>::kType);
    TensorWrapper pre_gelu(pre_gelu_buffer.get(),
                           {activation == 1 ? M * N * sizeof(ElementType) : 0},
                           NVTEWrapperDTypeMap<ElementType>::kType);

    return {std::move(a),    std::move(b),        std::move(c),
            std::move(bias), std::move(pre_gelu), std::move(workspace),
            accumulate};
  }

public:
  Status run(int64_t rankA, void *dstA, bool transa, int64_t rankB, void *dstB,
             bool transb, int64_t rankC, void *dstC, int64_t rankD, void *dstD,
             int64_t activation, float alpha, float beta) {

    auto [a, b, c, bias, pre_gelu, workspace, accumulate] =
        construct_tensors(rankA, dstA, transa, rankB, dstB, transb, rankC, dstC,
                          rankD, dstD, activation, alpha, beta);

    auto mpCount = getMulitProcessorCount();

    nvte_cublas_gemm(b.data(), a.data(), c.data(), bias.data(), pre_gelu.data(),
                     transb, transa, false, workspace.data(), accumulate, false,
                     mpCount, nullptr);

    auto error = cudaGetLastError();

    if (error != cudaSuccess) {
      return Status::kErrorInternal;
    }

    return Status::kSuccess;
  }
};

} // namespace homnvgpu_kernel
} // namespace hands_on_mlir
} // namespace mlir
