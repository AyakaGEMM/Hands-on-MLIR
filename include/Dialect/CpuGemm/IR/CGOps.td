#ifndef CG_CGDIALECT_TD
#define CG_CGDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// CG Dialect Definition.
//===----------------------------------------------------------------------===//

def CG_Dialect : Dialect {
  let name = "cg";
  let summary = "The CPU GEMM Dialect.";
  let description = [{
    The `cg` dialect is for optimizing cpu gemm.
  }];
  let cppNamespace = "::hands_on_mlir::cg";
}

//===----------------------------------------------------------------------===//
// Base CG Operation Definition.
//===----------------------------------------------------------------------===//

class CG_Op<string mnemonic, list<Trait> traits = []> :
    Op<CG_Dialect, mnemonic, traits>;

#endif // CG_CGDIALECT_TD