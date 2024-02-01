#ifndef HOM_TRANSFORMS_PASSES_H
#define HOM_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DECL_HOMFUSIONPASS
#define GEN_PASS_DECL_HOMSERIALIZEWEIGHTPASS
#define GEN_PASS_REGISTRATION
#include "HOM/Passes.h.inc"

} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir

#endif // HOM_TRANSFORMS_PASSES_H
