#ifndef HOM_HOMNVGPUDIALECT_TD
#define HOM_HOMNVGPUDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "HOM/HOMTypesBase.td" // Just reuses the hom types.

//===----------------------------------------------------------------------===//
// HOMNVGPU Dialect Definition.
//===----------------------------------------------------------------------===//

def HOMNVGPU_Dialect : Dialect {
  let name = "homnvgpu";
  let summary = "The Hands on MLIR Dialect for nvgpu.";
  let description =
      [{ The `HOMNVGPU` dialect is for better nv gpu related optimization. }];
  let cppNamespace = "::mlir::hands_on_mlir::homnvgpu";
}

class HOMNVGPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<HOMNVGPU_Dialect, mnemonic, !listconcat(traits, [Pure])>;

def HOMNVGPU_PrintOp : HOMNVGPU_Op<"print", []> {
  let arguments = (ins F64Tensor : $input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // HOM_HOMNVGPUDIALECT_TD
