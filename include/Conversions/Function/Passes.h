#ifndef HOM_CONVERSIONS_FUNC_TRANSFORMS_PASSES_H
#define HOM_CONVERSIONS_FUNC_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DECL_HOMTOFUNCPASS
#define GEN_PASS_DECL_UNIFYLLVMFUNCINTERFACEPASS
#define GEN_PASS_REGISTRATION
#include "Conversions/Function/Passes.h.inc"

} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir

#endif // HOM_CONVERSIONS_FUNC_TRANSFORMS_PASSES_H
