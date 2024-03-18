#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

#include "NVGPUKernels/OperationRunner.h"
#include "NVGPUKernels/Utils.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm_layernorm_mainloop_fusion.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/kernel/default_gemm_complex.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/util/device_layernorm.h"
#include "transformer_engine/layer_norm.h"
#include "transformer_engine/transformer_engine.h"

using MatrixCoord = cutlass::MatrixCoord;

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu_kernel {

// Compute stuff.
template <typename ElementVariance_, typename ElementMean_,
          typename ElementLayernormCompute_, typename ElementOutput,
          typename ThreadblockShape_, bool IsShiftedVariance_ = false>
class ApplyFinalReduction {
public:
  using ElementVariance = ElementVariance_;
  using ElementMean = ElementMean_;
  using ElementLayernormCompute = ElementLayernormCompute_;
  using ThreadblockShape = ThreadblockShape_;

  // Pre-processing has ensured the layout equivalent to RowMajor
  using Layout = cutlass::layout::RowMajor;

  using TensorVariance = cutlass::TensorRef<ElementVariance, Layout>;
  using TensorMean = cutlass::TensorRef<ElementMean, Layout>;

  static bool const kIsShiftedVariance = IsShiftedVariance_;

  //
  // Arguments
  //

  struct Arguments {

    MatrixCoord extent; ///< Extent of D and Layernorm matrices
    ElementVariance
        *ref_Variance;     ///< Sum Square or Variance tensor (input / output)
    ElementMean *ref_Mean; ///< Sum or Mean tensor (input / output)
    ElementOutput *ptr_Shifted_K; ///< Shifted K tensor pointer
    ElementLayernormCompute eps;

    //
    // Methods
    //
    Arguments() {}

    Arguments(MatrixCoord extent_, ElementVariance *ref_Variance_,
              ElementMean *ref_Mean_, ElementOutput *ptr_Shifted_K_,
              ElementLayernormCompute eps_ = 1e-5)
        : extent(extent_), ref_Variance(ref_Variance_), ref_Mean(ref_Mean_),
          ptr_Shifted_K(ptr_Shifted_K_), eps(eps_) {}
  };

  struct SharedStorage {};

  //
  // Params struct
  //

  struct Params {
    Arguments args;

    //
    // Methods
    //
    Params() {}

    Params(Arguments const &args_) : args(args_) {}
  };

private:
public:
  CUTLASS_DEVICE
  ApplyFinalReduction() {}

  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    apply(params, shared_storage);
  }

private:
  /// Partial reduction
  CUTLASS_DEVICE
  void apply(Params const &params, SharedStorage &shared_storage) {

    int threadblock_num =
        (params.args.extent.column() + ThreadblockShape::kM - 1) /
        ThreadblockShape::kM;

    int block_n = blockIdx.x * blockDim.x;

    int thread_n = threadIdx.x;

    int idx_n = block_n + thread_n;

    if (idx_n >= params.args.extent.row()) {
      return;
    }

    using ConvertVarianceOutput =
        cutlass::NumericConverter<ElementVariance, ElementLayernormCompute>;
    using ConvertMeanOutput =
        cutlass::NumericConverter<ElementMean, ElementLayernormCompute>;

    using ConvertVariance =
        cutlass::NumericConverter<ElementLayernormCompute, ElementVariance>;
    using ConvertMean =
        cutlass::NumericConverter<ElementLayernormCompute, ElementMean>;

    using ConvertShiftK =
        cutlass::NumericConverter<ElementLayernormCompute, ElementOutput>;

    ConvertVariance convert_variance;
    ConvertMean convert_mean;

    ConvertVarianceOutput convert_variance_output;
    ConvertMeanOutput convert_mean_output;

    ElementVariance *access_square = params.args.ref_Variance + idx_n;
    ElementMean *access_mean = params.args.ref_Mean + idx_n;

    ElementVariance *access_square_bak = access_square;
    ElementMean *access_mean_bak = access_mean;

    ElementLayernormCompute frag_square_sum = ElementLayernormCompute(0);
    ElementLayernormCompute frag_element_sum = ElementLayernormCompute(0);
    ElementVariance fetch_square;
    ElementMean fetch_mean;

    CUTLASS_PRAGMA_UNROLL
    for (int idx_m = 0; idx_m < threadblock_num; idx_m++) {
      cutlass::arch::global_load<ElementVariance, sizeof(ElementVariance)>(
          fetch_square, access_square, true);
      cutlass::arch::global_load<ElementMean, sizeof(ElementMean)>(
          fetch_mean, access_mean, true);
      frag_element_sum += convert_mean(fetch_mean);
      frag_square_sum += convert_variance(fetch_square);
      access_square += params.args.extent.row();
      access_mean += params.args.extent.row();
    }

    ElementLayernormCompute mean = frag_element_sum;
    ElementLayernormCompute square_mean = frag_square_sum;

    ElementLayernormCompute variance;

    if (kIsShiftedVariance && params.args.ptr_Shifted_K != nullptr) {
      ElementOutput *access_shift_k = params.args.ptr_Shifted_K + idx_n;
      ElementOutput fetch_shift_k;
      ConvertShiftK convert_shift_k;
      cutlass::arch::global_load<ElementOutput, sizeof(ElementOutput)>(
          fetch_shift_k, access_shift_k, true);
      ElementLayernormCompute shifted_mean =
          mean - convert_shift_k(fetch_shift_k);
      variance = cutlass::constants::one<ElementLayernormCompute>() /
                 cutlass::fast_sqrt(square_mean - shifted_mean * shifted_mean +
                                    params.args.eps);
    } else {
      variance =
          cutlass::constants::one<ElementLayernormCompute>() /
          cutlass::fast_sqrt(square_mean - mean * mean + params.args.eps);
    }

    mean = -mean * variance;

    access_square = access_square_bak;
    access_mean = access_mean_bak;

    access_square[0] = convert_variance_output(variance);
    access_mean[0] = convert_mean_output(mean);
  }
};

template <typename ThreadblockShape_, int ThreadCount,
          typename OutputTileIterator_, typename AccumulatorTile_,
          typename ElementAccumulator_, typename ElementVariance_,
          typename ElementMean_, typename ElementLayernormCompute_,
          typename ElementwiseFunctor_, bool IsShiftedVariance_ = false>
class EpilogueVisitorLayerNorm {
public:
  using ElementVariance = ElementVariance_;
  using ElementMean = ElementMean_;
  using ElementLayernormCompute = ElementLayernormCompute_;

  using AccumulatorTile = AccumulatorTile_;

  using ThreadblockShape = ThreadblockShape_;
  static int const kThreadCount = ThreadCount;

  using OutputTileIterator = OutputTileIterator_;
  using ElementwiseFunctor = ElementwiseFunctor_;

  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
  static int const kRowIterations =
      OutputTileIterator::ThreadMap::Iterations::kRow;

  static int const kThreads = OutputTileIterator::ThreadMap::kThreads;

  static bool const kIsShiftedVariance = IsShiftedVariance_;

  using ElementOutput = typename OutputTileIterator::Element;

  static int const kDeltaRow = OutputTileIterator::ThreadMap::Delta::kRow;

  /// Array type used in Shift-K Layernorm
  static int const kRowAccessCount = kIterations * kRowIterations;

  using ConvertedShiftFragment =
      cutlass::Array<ElementLayernormCompute, kRowAccessCount>;

  // Conducts manual transpose externally (already supported) for column major
  using LayoutOutput = cutlass::layout::RowMajor;

  using ElementAccumulator = ElementAccumulator_;

  using AccumulatorFragment =
      cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using LayernormFragment =
      cutlass::Array<ElementLayernormCompute, kElementsPerAccess>;
  using OutputVector = cutlass::Array<ElementOutput, kElementsPerAccess>;
  using TensorRefD = cutlass::TensorRef<ElementOutput, LayoutOutput>;

  static int const kThreadsPerRow = OutputTileIterator::ThreadMap::Detail::
      RowArrangement::Detail::kShapeWidth;
  static int const kThreadsInColumn = kThreads / kThreadsPerRow;
  static int const kHalfThreadsPerRow = (kThreadsPerRow >> 1);

  /// Argument structure
  struct Arguments {

    typename ElementwiseFunctor::Params elementwise;
    TensorRefD ref_C;
    TensorRefD ref_D;
    ElementVariance *ptr_Variance;
    ElementMean *ptr_Mean;
    ElementOutput *ptr_Shifted_K;

    //
    // Methods
    //
    Arguments()
        : ptr_Variance(nullptr), ptr_Mean(nullptr), ptr_Shifted_K(nullptr) {}

    Arguments(typename ElementwiseFunctor::Params elementwise_,
              TensorRefD ref_C_, TensorRefD ref_D_,
              ElementVariance *ptr_Variance, ElementMean *ptr_Mean_,
              ElementOutput *ptr_Shifted_K_ = nullptr)
        : elementwise(elementwise_), ref_C(ref_C_), ref_D(ref_D_),
          ptr_Variance(ptr_Variance), ptr_Mean(ptr_Mean_),
          ptr_Shifted_K(ptr_Shifted_K_) {}
  };

  struct Params {

    typename ElementwiseFunctor::Params elementwise;
    typename OutputTileIterator::Params params_C;
    typename OutputTileIterator::Params params_D;
    typename OutputTileIterator::Element *ptr_C;
    typename OutputTileIterator::Element *ptr_D;
    ElementVariance *ptr_Variance;
    ElementMean *ptr_Mean;
    ElementOutput *ptr_Shifted_K;

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() : ptr_D(nullptr), ptr_Variance(nullptr), ptr_Mean(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args)
        : elementwise(args.elementwise), params_C(args.ref_C.layout()),
          params_D(args.ref_D.layout()), ptr_C(args.ref_C.data()),
          ptr_D(args.ref_D.data()), ptr_Variance(args.ptr_Variance),
          ptr_Mean(args.ptr_Mean), ptr_Shifted_K(args.ptr_Shifted_K) {}
  };

  /// Shared storage
  struct SharedStorage {};

private:
  Params const &params_;
  SharedStorage &shared_storage_;
  MatrixCoord extent_;
  ElementwiseFunctor elementwise_;

  OutputTileIterator iterator_C_;
  OutputTileIterator iterator_D_;
  typename OutputTileIterator::Fragment fragment_C_;
  typename OutputTileIterator::Fragment fragment_D_;

  ElementAccumulator alpha_;
  ElementAccumulator beta_;
  ConvertedShiftFragment shift_k_frag_;

  ElementLayernormCompute accum_sum_square_;
  ElementLayernormCompute accum_sum_element_;

  MatrixCoord thread_offset_;

public:
  CUTLASS_DEVICE
  EpilogueVisitorLayerNorm(
      Params const &params, ///< Parameters routed to the epilogue
      SharedStorage
          &shared_storage, ///< Shared storage needed by the functors here
      MatrixCoord const &problem_size0, ///< Problem size of the output
      int thread_idx,                   ///< Thread index within the threadblock
      int warp_idx,                     ///< Warp index within the threadblock
      int lane_idx,                     ///< Lane index within the warp
      MatrixCoord const &threadblock_offset = MatrixCoord(0, 0))
      : params_(params), shared_storage_(shared_storage),
        extent_(problem_size0), elementwise_(params.elementwise),
        iterator_C_(params.params_C, params.ptr_C, problem_size0, thread_idx,
                    threadblock_offset),
        iterator_D_(params.params_D, params.ptr_D, problem_size0, thread_idx,
                    threadblock_offset) {
    alpha_ = (params.elementwise.alpha_ptr ? *params.elementwise.alpha_ptr
                                           : params.elementwise.alpha);
    beta_ = (params.elementwise.beta_ptr ? *params.elementwise.beta_ptr
                                         : params.elementwise.beta);

    if (beta_ == ElementAccumulator()) {
      iterator_C_.clear_mask();
    }
  }

  /// Helper to indicate split-K behavior
  CUTLASS_DEVICE
  void set_k_partition(int split_k_index, ///< Index of this threadblock within
                                          ///< split-K partitioned scheme
                       int split_k_slices) { ///< Total number of split-K slices
  }

  /// Called to set the batch index
  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) {}

  /// Called at the start of the epilogue just before iterating over accumulator
  /// slices
  CUTLASS_DEVICE
  void begin_epilogue() {

    // If shift-K feature is enabled, we load shift-k fragment
    // at the very beginning of an epilogue
    if (kIsShiftedVariance && params_.ptr_Shifted_K != nullptr) {
      shift_k_frag_.clear();
      int thread_offset_row_base = iterator_D_.thread_start_row();

      CUTLASS_PRAGMA_UNROLL
      for (int iter_idx = 0; iter_idx < kIterations; ++iter_idx) {
        int step_offset = iter_idx * OutputTileIterator::Shape::kRow;
        CUTLASS_PRAGMA_UNROLL
        for (int rid = 0; rid < kRowIterations; ++rid) {
          int row_step_offset = rid * kDeltaRow;
          int row_offset =
              thread_offset_row_base + step_offset + row_step_offset;
          bool is_load = (row_offset < extent_.row());
          shift_k_frag_[iter_idx * kRowIterations + rid] =
              load_shift_k_(row_offset, is_load);
        }
      }
    }
  }

  /// Called at the start of one step before starting accumulator exchange
  CUTLASS_DEVICE
  void begin_step(int step_idx) {
    fragment_D_.clear();

    if (elementwise_.kScale !=
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
      fragment_C_.clear();
      iterator_C_.load(fragment_C_);
      ++iterator_C_;
    }
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void begin_row(int row_idx) {}

  /// Called after accumulators have been exchanged for each accumulator vector
  CUTLASS_DEVICE
  void visit(int iter_idx, int row_idx, int column_idx, int frag_idx,
             AccumulatorFragment const &accum) {

    using Mul = cutlass::multiplies<ElementLayernormCompute>;
    using Minus = cutlass::minus<ElementLayernormCompute>;
    using Exp = cutlass::fast_exp_op<ElementLayernormCompute>;

    [[maybe_unused]] Minus minus;
    [[maybe_unused]] Mul mul;
    [[maybe_unused]] Exp exponential;

    LayernormFragment result;

    thread_offset_ = iterator_D_.thread_start() +
                     OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

    cutlass::NumericArrayConverter<ElementLayernormCompute, ElementOutput,
                                   kElementsPerAccess>
        source_converter;
    OutputVector &source_vector =
        reinterpret_cast<OutputVector *>(&fragment_C_)[frag_idx];

    bool column_guard = (thread_offset_.column() < extent_.column());

    if (elementwise_.kScale ==
        cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling) {
      result = source_converter(elementwise_(accum));
    } else {
      result = source_converter(elementwise_(accum, source_vector));
    }

    ElementLayernormCompute inv_scalar =
        cutlass::constants::one<ElementLayernormCompute>() /
        ElementLayernormCompute(extent_.column());

    // Fragment is cleared for non-reachable columns so no need to check against
    // column guard
    accum_sum_element_ = element_sum_accumulator_(result);

    // Square sum is different. Non-reachable columns should've been computed
    // for shift-k Otherwise we will incorrectly have some extra k^2 added into
    // square sum.
    if (column_guard) {
      accum_sum_square_ =
          (kIsShiftedVariance)
              ? square_sum_accumulator_(
                    result, shift_k_frag_[iter_idx * kRowIterations + row_idx])
              : square_sum_accumulator_(result);
    } else {
      accum_sum_square_ = ElementLayernormCompute(0);
    }

    accum_sum_element_ *= inv_scalar;
    accum_sum_square_ *= inv_scalar;

    // After performing the in-thread reduction, we then perform cross-thread /
    // in-warp reduction
    CUTLASS_PRAGMA_UNROLL
    for (int i = kHalfThreadsPerRow; i > 0; i >>= 1) {
      accum_sum_element_ += __shfl_xor_sync(0xFFFFFFFF, accum_sum_element_, i);
      accum_sum_square_ += __shfl_xor_sync(0xFFFFFFFF, accum_sum_square_, i);
    }

    // Convert to the output
    cutlass::NumericArrayConverter<ElementOutput, ElementLayernormCompute,
                                   kElementsPerAccess>
        output_converter;
    OutputVector &output =
        reinterpret_cast<OutputVector *>(&fragment_D_)[frag_idx];
    output = output_converter(result);
  }

  /// Called at the start of a row
  CUTLASS_DEVICE
  void end_row(int row_idx) {

    using ConvertVarianceOutput =
        cutlass::NumericConverter<ElementVariance, ElementLayernormCompute>;
    using ConvertMeanOutput =
        cutlass::NumericConverter<ElementMean, ElementLayernormCompute>;

    ConvertVarianceOutput convert_variance_output;
    ConvertMeanOutput convert_mean_output;

    bool is_write_thread = (thread_offset_.row() < extent_.row() &&
                            (threadIdx.x % kThreadsPerRow) == 0);
    int row_offset = thread_offset_.row() + blockIdx.y * extent_.row();

    ElementVariance *curr_ptr_sum_square = params_.ptr_Variance + row_offset;
    ElementMean *curr_ptr_element_sum = params_.ptr_Mean + row_offset;

    cutlass::arch::global_store<ElementVariance, sizeof(ElementVariance)>(
        convert_variance_output(accum_sum_square_), (void *)curr_ptr_sum_square,
        is_write_thread);

    cutlass::arch::global_store<ElementMean, sizeof(ElementMean)>(
        convert_mean_output(accum_sum_element_), (void *)curr_ptr_element_sum,
        is_write_thread);
  }

  /// Called after all accumulator elements have been visited
  CUTLASS_DEVICE
  void end_step(int step_idx) {

    iterator_D_.store(fragment_D_);
    ++iterator_D_;
  }

  /// Called after all steps have been completed
  CUTLASS_DEVICE
  void end_epilogue() {}

private:
  CUTLASS_DEVICE
  ElementLayernormCompute load_shift_k_(int row_offset, bool is_load) {
    using ConvertShiftK =
        cutlass::NumericConverter<ElementLayernormCompute, ElementOutput>;
    ConvertShiftK convert_shift_k;
    ElementOutput shift_k_val;

    // Computes the address to load shift_k element
    ElementOutput *curr_ptr_shift_k = params_.ptr_Shifted_K + row_offset;
    // Conditionally loads from global memory
    cutlass::arch::global_load<ElementOutput, sizeof(ElementOutput)>(
        shift_k_val, (void *)curr_ptr_shift_k, is_load);
    // Converts data type to return
    ElementLayernormCompute converted_shift_k_val =
        convert_shift_k(shift_k_val);

    return converted_shift_k_val;
  }

  CUTLASS_DEVICE
  ElementLayernormCompute
  square_sum_accumulator_(LayernormFragment const &accum) {
    ElementLayernormCompute sum_ = ElementLayernormCompute(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < LayernormFragment::kElements; ++i) {
      auto accum_ = accum[i];
      sum_ += accum_ * accum_;
    }

    return sum_;
  }

  CUTLASS_DEVICE
  ElementLayernormCompute
  square_sum_accumulator_(LayernormFragment const &accum,
                          ElementLayernormCompute shift_k_val) {
    ElementLayernormCompute sum_ = ElementLayernormCompute(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < LayernormFragment::kElements; ++i) {
      auto accum_ = accum[i] - shift_k_val;
      sum_ += accum_ * accum_;
    }

    return sum_;
  }

  CUTLASS_DEVICE
  ElementLayernormCompute
  element_sum_accumulator_(LayernormFragment const &accum) {
    ElementLayernormCompute sum_ = ElementLayernormCompute(0);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < LayernormFragment::kElements; ++i) {
      sum_ += accum[i];
    }

    return sum_;
  }
};

template <typename ElementType> class LayernormRunner : public OperationRunner {

  using TensorWrapper = transformer_engine::TensorWrapper;

  static void getWorkSpace(size_t bs, size_t hidden_size, float eps,
                           TensorWrapper &workspace, TensorWrapper &barrier,
                           int &mpCount) {
    auto input_shape = std::vector<size_t>{bs, hidden_size};
    auto weight_shape = std::vector<size_t>{hidden_size};
    auto intermediates_shape = std::vector<size_t>{bs};

    auto input_tensor = TensorWrapper(nullptr, input_shape,
                                      NVTEWrapperDTypeMap<ElementType>::kType);
    auto gamma_tensor = TensorWrapper(nullptr, weight_shape,
                                      NVTEWrapperDTypeMap<ElementType>::kType);
    auto beta_tensor = TensorWrapper(nullptr, weight_shape,
                                     NVTEWrapperDTypeMap<ElementType>::kType);
    auto output_tensor = TensorWrapper(nullptr, input_shape,
                                       NVTEWrapperDTypeMap<ElementType>::kType);
    auto rsigma_tensor = TensorWrapper(nullptr, intermediates_shape,
                                       NVTEWrapperDTypeMap<float>::kType);
    auto mu_tensor = TensorWrapper(nullptr, intermediates_shape,
                                   NVTEWrapperDTypeMap<float>::kType);

    mpCount = getMulitProcessorCount();

    nvte_layernorm1p_fwd(input_tensor.data(), gamma_tensor.data(),
                         beta_tensor.data(), eps, output_tensor.data(),
                         mu_tensor.data(), rsigma_tensor.data(), nullptr,
                         mpCount, workspace.data(), barrier.data());
  }

  std::tuple<TensorWrapper, TensorWrapper, TensorWrapper, TensorWrapper,
             TensorWrapper, TensorWrapper, TensorWrapper, TensorWrapper, int>
  construct_tensors(int64_t rankA, void *dstA, float eps) {
    auto A = convertToDynamicMemRefType<ElementType>(rankA, dstA);

    assert(A.rank == 3 || A.rank == 2);

    size_t bs = A.rank == 3 ? A.sizes[0] * A.sizes[1] : A.sizes[0];
    size_t hidden_size = A.rank == 3 ? A.sizes[2] : A.sizes[1];

    auto zeroData = getZeroPointer<ElementType>(hidden_size * 2);
    auto outData = getDummyPointer<float>(bs * 2);

    TensorWrapper workspace, barrier;
    int mpCount;

    getWorkSpace(bs, hidden_size, eps, workspace, barrier, mpCount);

    size_t workspaceSize = workspace.shape().data[0] *
                           getNVTEWrapperDTypeSize(workspace.dtype()),
           barrierSize = barrier.shape().data[0] *
                         getNVTEWrapperDTypeSize(barrier.dtype());

    auto workspace_buffer = getDummyPointer(workspaceSize + barrierSize);

    workspace = TensorWrapper(workspace_buffer.get(), workspace.shape(),
                              workspace.dtype());
    barrier = TensorWrapper(workspace_buffer.get() + workspaceSize,
                            barrier.shape(), barrier.dtype());

    auto input_shape = std::vector<size_t>{bs, hidden_size};
    auto weight_shape = std::vector<size_t>{hidden_size};
    auto intermediates_shape = std::vector<size_t>{bs};

    auto input_tensor = TensorWrapper(A.data, input_shape,
                                      NVTEWrapperDTypeMap<ElementType>::kType);
    auto gamma_tensor = TensorWrapper(zeroData.get(), weight_shape,
                                      NVTEWrapperDTypeMap<ElementType>::kType);
    auto beta_tensor = TensorWrapper(zeroData.get() + hidden_size, weight_shape,
                                     NVTEWrapperDTypeMap<ElementType>::kType);
    auto output_tensor = TensorWrapper(A.data, input_shape,
                                       NVTEWrapperDTypeMap<ElementType>::kType);
    auto rsigma_tensor = TensorWrapper(outData.get(), intermediates_shape,
                                       NVTEWrapperDTypeMap<float>::kType);
    auto mu_tensor = TensorWrapper(outData.get() + bs, intermediates_shape,
                                   NVTEWrapperDTypeMap<float>::kType);

    return {std::move(input_tensor),
            std::move(gamma_tensor),
            std::move(beta_tensor),
            std::move(output_tensor),
            std::move(rsigma_tensor),
            std::move(mu_tensor),
            std::move(workspace),
            std::move(barrier),
            mpCount};
  }

public:
  Status run(int64_t rankA, void *dstA, float eps) {

    auto [inputTensor, gammaTensor, betaTensor, outputTensor, rsigmaTensor,
          muTensor, workspace, barrier, mpCount] =
        construct_tensors(rankA, dstA, eps);

    checkCudaErrors(cudaStreamSynchronize(nullptr));

    nvte_layernorm1p_fwd(inputTensor.data(), gammaTensor.data(),
                         betaTensor.data(), eps, outputTensor.data(),
                         muTensor.data(), rsigmaTensor.data(), nullptr, mpCount,
                         workspace.data(), barrier.data());

    checkCudaErrors(cudaStreamSynchronize(nullptr));

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
