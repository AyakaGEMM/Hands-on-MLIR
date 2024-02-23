#ifndef HOM_CONVERSIONS_FP32TOFP16_PASS_H_
#define HOM_CONVERSIONS_FP32TOFP16_PASS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace hands_on_mlir {

#define GEN_PASS_DECL_HOMFP32TOFP16PASS
#define GEN_PASS_REGISTRATION
#include "Conversions/FP32toFP16/Passes.h.inc"

} // namespace hands_on_mlir
} // namespace mlir

#endif
