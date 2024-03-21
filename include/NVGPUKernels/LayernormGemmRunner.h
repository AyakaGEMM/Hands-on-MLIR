#pragma once

#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/Layernorm.h"
#include "NVGPUKernels/OperationRunner.h"
#include "NVGPUKernels/Utils.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "transformer_engine/gemm.h"
#include "transformer_engine/transformer_engine.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu_kernel {

template <typename Kernel_, bool IsShiftedVariance_ = false>
class GemmOperationWithVarMeanRunner {
public:
  using Kernel = Kernel_;
  using ElementA = typename Kernel::ElementA;
  using LayoutA = typename Kernel::LayoutA;
  using ElementB = typename Kernel::ElementB;
  using LayoutB = typename Kernel::LayoutB;
  using ElementC = typename Kernel::ElementC;
  using LayoutC = typename Kernel::LayoutC;
  using ElementD = ElementC;
  using LayoutD = LayoutC;
  // assuming all tensors use same type for StrideIndex
  using StrideIndex = typename Kernel::LayoutA::Index;

  using KernelArguments = typename Kernel::Arguments;

  using ElementLayernormCompute = float;
  using ElementVariance = ElementD;
  using ElementMean = ElementD;

  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  static bool const kInternalTranspose =
      cutlass::platform::is_same<LayoutC, cutlass::layout::ColumnMajor>::value;
  static bool const kIsShiftedVariance = IsShiftedVariance_;

  using ApplyFinalReductionKernel =
      ApplyFinalReduction<ElementVariance, ElementMean, ElementLayernormCompute,
                          ElementC, typename Kernel::ThreadblockShape,
                          kIsShiftedVariance>;

  struct Arguments {
    KernelArguments gemm;
    typename ApplyFinalReductionKernel::Arguments reduction;
    cutlass::gemm::GemmCoord extend;

    Arguments() {}

    Arguments(cutlass::gemm::GemmCoord problem_size, ElementA *ptrA,
              ElementB *ptrB, ElementC *ptrC, ElementD *ptrD,
              ElementVariance *ptrVar, ElementMean *ptrMean, int64_t lda,
              int64_t ldb, int64_t ldc, int64_t ldd,
              typename Kernel::EpilogueVisitor::ElementwiseFunctor::Params
                  linear_scaling,
              ElementLayernormCompute eps)
        : gemm(cutlass::gemm::GemmUniversalMode::kGemm,
               {kInternalTranspose ? problem_size.n() : problem_size.m(),
                kInternalTranspose ? problem_size.m() : problem_size.n(),
                problem_size.k()},
               {kInternalTranspose ? ptrB : ptrA,
                kInternalTranspose ? ldb : lda},
               {kInternalTranspose ? ptrA : ptrB,
                kInternalTranspose ? lda : ldb},
               typename Kernel_::EpilogueVisitor::Arguments(
                   linear_scaling, {ptrC, ldc}, {ptrD, ldd}, ptrVar, ptrMean,
                   nullptr)),
          reduction(MatrixCoord(kInternalTranspose ? problem_size.n()
                                                   : problem_size.m(),
                                kInternalTranspose ? problem_size.m()
                                                   : problem_size.n()),
                    ptrVar, ptrMean, nullptr, eps),
          extend(problem_size) {}
  };

  struct Params {
    typename Kernel::Params gemm;
    typename ApplyFinalReductionKernel::Params reduction;
    MatrixCoord extend;

    Params() {}

    Params(const Arguments &args)
        : gemm(args.gemm), reduction(args.reduction),
          extend(MatrixCoord(args.extend.m(), args.extend.n())) {}
  };

private:
  Params params_;

public:
  /// Constructor
  GemmOperationWithVarMeanRunner(char const *name = "unknown_gemm") {}

  ~GemmOperationWithVarMeanRunner() {}

  Status construct_arguments(int64_t rankA, void *dstA, int64_t rankB,
                             void *dstB, int64_t rankC, void *dstC,
                             int64_t rankD, void *dstD, int64_t rankVar,
                             void *dstVar, int64_t rankMean, void *dstMean,
                             float alpha, float beta, float eps,
                             Arguments &args) {
    auto A = convertToDynamicMemRefType<ElementA>(rankA, dstA);
    auto B = convertToDynamicMemRefType<ElementB>(rankB, dstB);
    auto C = convertToDynamicMemRefType<ElementC>(rankC, dstC);
    auto D = convertToDynamicMemRefType<ElementD>(rankD, dstD);
    auto Mean = convertToDynamicMemRefType<ElementMean>(rankMean, dstMean);
    auto Var = convertToDynamicMemRefType<ElementVariance>(rankVar, dstVar);

    assert(A.rank == 3);
    assert(B.rank == 3);
    assert(C.rank == 3);
    assert(D.rank == 3);
    assert(Mean.rank == 1);
    assert(Var.rank == 1);

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

    args = Arguments({int(M * BatchSize), int(N), int(K)}, A.data, B.data,
                     C.data, D.data, Var.data, Mean.data, A.sizes[2],
                     B.sizes[2], C.sizes[2], D.sizes[2],
                     {ElementC(alpha), ElementD(beta)}, eps);

    params_ = Params(args);

    return Status::kSuccess;
  }

  Status run(int64_t rankA, void *dstA, int64_t rankB, void *dstB,
             int64_t rankC, void *dstC, int64_t rankD, void *dstD,
             int64_t rankVar, void *dstVar, int64_t rankMean, void *dstMean,
             float alpha, float beta, float eps) {

    Arguments args;
    auto status = construct_arguments(rankA, dstA, rankB, dstB, rankC, dstC,
                                      rankD, dstD, rankVar, dstVar, rankMean,
                                      dstMean, alpha, beta, eps, args);
    if (status != Status::kSuccess) {
      return status;
    }

    //
    // Launch the GEMM + layernorm kernel
    //

    dim3 gemm_grid =
        SwizzleThreadBlock().get_grid_shape(params_.gemm.grid_tiled_shape);
    dim3 gemm_block(Kernel::kThreadCount, 1, 1);

    int gemm_smem_size = int(sizeof(typename Kernel::SharedStorage));

    cutlass::Kernel<Kernel>
        <<<gemm_grid, gemm_block, gemm_smem_size>>>(params_.gemm);

    checkCudaErrors(cudaGetLastError());

    cudaError_t result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // Launch the ApplyFinalReductionKernel
    //

    // always performs reduction from leading dimension
    int leading_dim_0 =
        kInternalTranspose ? params_.extend.row() : params_.extend.column();
    int leading_dim_1 =
        kInternalTranspose ? params_.extend.column() : params_.extend.row();

    int thread_per_block = 128;
    int block_per_row =
        (leading_dim_1 + thread_per_block - 1) / thread_per_block;
    if (block_per_row < 4) {
      thread_per_block = 32;
      block_per_row = (leading_dim_1 + thread_per_block - 1) / thread_per_block;
    }

    dim3 final_reduction_block(thread_per_block);
    dim3 final_reduction_grid(block_per_row);

    cutlass::Kernel<ApplyFinalReductionKernel>
        <<<final_reduction_grid, final_reduction_block,
           sizeof(typename ApplyFinalReductionKernel::SharedStorage)>>>(
            params_.reduction);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    return Status::kSuccess;
  }
};

template <typename Operator_>
class LayernormGemmOperationRunner : public OperationRunner {
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

  static bool const kInternalTranspose =
      cutlass::platform::is_same<LayoutC, cutlass::layout::ColumnMajor>::value;

  LayernormGemmOperationRunner() {
    host_workspace_ = device_workspace_ = nullptr;
    device_workspace_size_ = 0;
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
                                    int64_t rankD, void *dstD, int64_t rankVar,
                                    void *dstVar, int64_t rankMean,
                                    void *dstMean, float alpha, float beta,
                                    OperatorArguments &args) {
    auto A = convertToDynamicMemRefType<ElementA>(rankA, dstA);
    auto B = convertToDynamicMemRefType<ElementB>(rankB, dstB);
    auto C = convertToDynamicMemRefType<ElementC>(rankC, dstC);
    auto D = convertToDynamicMemRefType<ElementD>(rankD, dstD);
    auto Var = convertToDynamicMemRefType<ElementD>(rankVar, dstVar);
    auto Mean = convertToDynamicMemRefType<ElementD>(rankMean, dstMean);

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

    assert(B.sizes[0] == 1);
    assert(BatchSize == C.sizes[0]);
    assert(BatchSize == D.sizes[0]);
    assert(C.sizes[0] == D.sizes[0]);
    assert(C.sizes[1] == D.sizes[1]);
    assert(C.sizes[2] == D.sizes[2]);

    auto gamma = getOnePointer<ElementA>(K);
    auto ln_beta = getZeroPointer<ElementA>(K);

    args = OperatorArguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {int(M * BatchSize), int(N), int(K)}, 1,
        {ElementC(alpha), ElementC(beta)}, kInternalTranspose ? B.data : A.data,
        kInternalTranspose ? A.data : B.data, Var.data, Mean.data, gamma.get(),
        ln_beta.get(), C.data, D.data, M * BatchSize * K, N * K, M * BatchSize,
        M * BatchSize, K, K, BatchSize * M * N, BatchSize * M * N,
        kInternalTranspose ? N : K, N, M * BatchSize, M * BatchSize, K, K, N,
        N);

    return Status::kSuccess;
  }

  Status run(int64_t rankA, void *dstA, int64_t rankB, void *dstB,
             int64_t rankC, void *dstC, int64_t rankD, void *dstD,
             int64_t rankVar, void *dstVar, int64_t rankMean, void *dstMean,
             float alpha, float beta) {

    OperatorArguments args;

    auto status = construct_arguments(rankA, dstA, rankB, dstB, rankC, dstC,
                                      rankD, dstD, rankVar, dstVar, rankMean,
                                      dstMean, alpha, beta, args);
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

protected:
  void *host_workspace_;
  void *device_workspace_;
  size_t device_workspace_size_;

public:
  Status run() { return Status::kSuccess; }
};

} // namespace homnvgpu_kernel
} // namespace hands_on_mlir
} // namespace mlir
