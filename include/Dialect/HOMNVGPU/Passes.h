#ifndef HOMNVGPU_TRANSFORMS_PASSES_H
#define HOMNVGPU_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu {

#define GEN_PASS_DECL_HOMNVGPUFUSIONPASS
#define GEN_PASS_DECL_HOMNVGPULEGALIZEGEMMPASS
#define GEN_PASS_REGISTRATION
#include "HOMNVGPU/Passes.h.inc"

} // namespace homnvgpu
} // namespace hands_on_mlir
} // namespace mlir

#endif // HOMNVGPU_TRANSFORMS_PASSES_H
