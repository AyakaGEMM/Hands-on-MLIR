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

class OperationRunner {
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

} // namespace hands_on_mlir
} // namespace mlir
