#ifndef HOM_CONVERSIONS_HOM_TRANSFORMS_PASSES_H
#define HOM_CONVERSIONS_HOM_TRANSFORMS_PASSES_H

#include "Dialect/HOM/HOMOps.h"
#include "Dialect/HOMNVGPU/HOMNVGPUOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hands_on_mlir {

#define GEN_PASS_DECL_HOMTOHOMNVGPUPASS
#define GEN_PASS_REGISTRATION
#include "Conversions/HOM/Passes.h.inc"

} // namespace hands_on_mlir
} // namespace mlir

#endif // HOM_CONVERSIONS_HOM_TRANSFORMS_PASSES_H
