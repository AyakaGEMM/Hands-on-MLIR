#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

using cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nn_align2_base =
    typename cutlass::gemm::kernel::DefaultGemmUniversal<
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone, 2, // transposed B operand
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone, 2, // transposed A operand
        cutlass::half_t, cutlass::layout::RowMajor, float,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<256, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,

        cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 2, float,
                                                     float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 2,
        cutlass::arch::OpMultiplyAdd>::GemmKernel;

// Define named type
struct cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nn_align2
    : public cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nn_align2_base {};

int main() {
  int64_t m = 64, n = 768, k = 768;

  cutlass::gemm::device::GemmUniversal<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t,
      cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::RowMajor,
      float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<256, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 8>,
      cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 2, float,
                                                   float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 2, 2, 2,
      cutlass::arch::OpMultiplyAdd>
      gemm;

  half *a, *b, *c, *d;
  cudaMalloc(&a, sizeof(half) * m * k);
  cudaMalloc(&b, sizeof(half) * n * k);
  cudaMalloc(&c, sizeof(half) * m * n);
  cudaMalloc(&d, sizeof(half) * m * n);

  cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nn_align2_base::Arguments
      args(cutlass::gemm::GemmUniversalMode::kBatched, {int(m), int(n), int(k)},
           1, {1.0f, 0.0f}, a, b, c, c, m * k, n * k, m * n, m * n, k, n, n, n);

  decltype(gemm)::Arguments argsGemm(
      cutlass::gemm::GemmUniversalMode::kGemm, {int(m), int(n), int(k)}, 1,
      {1.0, 1.0}, a, b, c, d, m * k, n * k, m * n, m * n, k, n, n, n);

  auto res = gemm.can_implement(argsGemm);
  if (res != cutlass::Status::kSuccess) {
    std::cout << "Not good" << std::endl;
  }

  res = gemm.initialize(argsGemm);

  if (res != cutlass::Status::kSuccess) {
    std::cout << "Not good" << std::endl;
  }

  std::cout << "Before run." << std::endl;

  gemm(argsGemm);

  res = gemm.run();

  auto err = cudaStreamSynchronize(nullptr);

  if (err != cudaSuccess) {
    std::cout << "A: " << cudaGetErrorString(err) << std::endl;
  }
}
