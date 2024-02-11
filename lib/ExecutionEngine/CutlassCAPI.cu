#include <stddef.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/half.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;
// Copy examples from cutlass here just to make sure we can really compile
// cutlass.
auto cutlassMatmul(int M, int N, int K, float alpha, float const *A, int lda,
                   float const *B, int ldb, float beta, float *C, int ldc) {

  // Define type definition for single-precision CUTLASS GEMM with
  // column-major input matrices and 128x128x8 threadblock tile size (chosen
  // by default).
  //
  // To keep the interface manageable, several helpers are defined for
  // plausible compositions including the following example for
  // single-precision GEMM. Typical values are used as default template
  // arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for
  // more details.
  //
  // To view the full gemm device API interface, see
  // `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm =
      cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                  ColumnMajor,  // Layout of A matrix
                                  float,        // Data-type of B matrix
                                  ColumnMajor,  // Layout of B matrix
                                  float,        // Data-type of C matrix
                                  ColumnMajor>; // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that
  // are constructible in host code and passed to kernels by value. These may
  // include pointers, strides, scalars, and other arguments needed by Gemm
  // and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy
  // for passing host-constructible arguments to kernels and (2.) minimized
  // initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args(
      {M, N, K},      // Gemm Problem dimensions
      {A, lda},       // Tensor-ref for source matrix A
      {B, ldb},       // Tensor-ref for source matrix B
      {C, ldc},       // Tensor-ref for source matrix C
      {C, ldc},       // Tensor-ref for destination matrix D (may be different
                      // memory than source C matrix)
      {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //

  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}
