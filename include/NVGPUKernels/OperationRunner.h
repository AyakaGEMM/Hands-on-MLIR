#pragma once
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/Utils.h"
#include "cutlass/coord.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/library/types.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace mlir {
namespace hands_on_mlir {

class Operation {
public:
};

struct MathInstructionDescription {

  using NumericTypeID = cutlass::library::NumericTypeID;
  using OpcodeClassID = cutlass::library::OpcodeClassID;
  using MathOperationID = cutlass::library::MathOperationID;

  /// Shape of the target math instruction
  cutlass::gemm::GemmCoord instruction_shape;

  /// Describes the data type of the internal accumulator
  NumericTypeID element_accumulator;

  /// Classification of math instruction
  OpcodeClassID opcode_class;

  /// Type of math operation performed
  MathOperationID math_operation;

  //
  // Methods
  //

  MathInstructionDescription(
      cutlass::gemm::GemmCoord instruction_shape = cutlass::gemm::GemmCoord(),
      NumericTypeID element_accumulator = NumericTypeID::kInvalid,
      OpcodeClassID opcode_class = OpcodeClassID::kInvalid,
      MathOperationID math_operation = MathOperationID::kMultiplyAdd)
      : instruction_shape(instruction_shape),
        element_accumulator(element_accumulator), opcode_class(opcode_class),
        math_operation(math_operation) {}

  // Equality operator
  inline bool operator==(MathInstructionDescription const &rhs) const {
    return ((instruction_shape == rhs.instruction_shape) &&
            (element_accumulator == rhs.element_accumulator) &&
            (opcode_class == rhs.opcode_class) &&
            (math_operation == rhs.math_operation));
  }

  // Inequality operator
  inline bool operator!=(MathInstructionDescription const &rhs) const {
    return !(*this == rhs);
  }
};

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

struct GemmDescription {
  const char *name;
  cutlass::library::Provider provider;
  cutlass::library::OperationKind kind;
  cutlass::library::GemmKind gemm_kind;
  TileDescription tile_description;
};

template <typename Operator_> class GemmOperationRunner : public Operation {
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
  GemmDescription description_;
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
  }

  ~GemmOperationRunner() {
    if (host_workspace_ != nullptr) {
      auto op = static_cast<Operator *>(host_workspace_);
      delete op;
    }

    if (device_workspace_size_ > 0) {
      cudaFree(device_workspace_);
    }
  }

  cutlass::Status initialize_and_update(const OperatorArguments &args) {
    bool first_initialize = false;
    if (host_workspace_ == nullptr) {
      host_workspace_ = new Operator;
      first_initialize = true;
    }
    Operator *op = static_cast<Operator *>(host_workspace_);
    auto status = op->can_implement(args);

    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    auto required_workspace = op->get_workspace_size(args);

    if (required_workspace > device_workspace_size_) {
      if (device_workspace_ != 0) {
        cudaFree(device_workspace_);
      }
      auto error = cudaMalloc(&device_workspace_, required_workspace);
      assert(error == cudaSuccess);
      device_workspace_size_ = required_workspace;
    }

    if (first_initialize) {
      return op->initialize(args, device_workspace_);
    } else {
      return op->update(args, device_workspace_);
    }
  }

  static cutlass::Status
  construct_arguments(int64_t rankA, void *dstA, int64_t rankB, void *dstB,
                      int64_t rankC, void *dstC, int64_t rankD, void *dstD,
                      float alpha, float beta, OperatorArguments &args) {
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

    assert(BatchSize == B.sizes[0]);
    assert(BatchSize == C.sizes[0]);
    assert(BatchSize == D.sizes[0]);
    assert(C.sizes[0] == D.sizes[0]);
    assert(C.sizes[1] == D.sizes[1]);
    assert(C.sizes[2] == D.sizes[2]);

    args.ref_A = {A.data, A.sizes[2]};
    args.ref_B = {B.data, B.sizes[2]};
    args.ref_C = {C.data, C.sizes[2]};
    args.ref_D = {D.data, D.sizes[2]};
    args.problem_size = {int(M), int(N), int(K)};
    args.epilogue = {alpha, beta};
    args.split_k_slices = 1;

    return cutlass::Status::kSuccess;
  }

  cutlass::Status run(int64_t rankA, void *dstA, int64_t rankB, void *dstB,
                      int64_t rankC, void *dstC, int64_t rankD, void *dstD,
                      float alpha, float beta) {

    OperatorArguments args;

    auto status = construct_arguments(rankA, dstA, rankB, dstB, rankC, dstC,
                                      rankD, dstD, alpha, beta, args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    status = initialize_and_update(args);
    if (status != cutlass::Status::kSuccess) {
      return status;
    }

    auto op = static_cast<Operator *>(host_workspace_);

    std::cout << "Run" << std::endl;

    return op->run();
  }
};
} // namespace hands_on_mlir
} // namespace mlir
