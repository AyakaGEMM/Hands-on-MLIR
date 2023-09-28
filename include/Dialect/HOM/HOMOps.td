#ifndef HOM_HOMDIALECT_TD
#define HOM_HOMDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// HOM Dialect Definition.
//===----------------------------------------------------------------------===//

def HOM_Dialect : Dialect {
  let name = "hom";
  let summary = "The Hands on MLIR Main Dialect.";
  let description = [{
    The `HOM` dialect is for better op management.
  }];
  let cppNamespace = "::mlir::hands_on_mlir::hom";
}

//===----------------------------------------------------------------------===//
// Base HOM Operation Definition.
//===----------------------------------------------------------------------===//

class HOM_Op<string mnemonic, list<Trait> traits = []> :
    Op<HOM_Dialect, mnemonic, traits>;

def HOM_PrintOp : HOM_Op<"print", []>{
  let arguments = (ins F64Tensor:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def HOM_ConstantOp : HOM_Op<"constant",
    [ConstantLike, Pure]> {
  let summary = "Constant operation";
  let description = [{
    Produces an `output` tensor from a constant `value`.

    Example:
    ```mlir
    %output = hom.constant dense<[[0.0, 1.0], [2.0, 3.0]]> : tensor<2x2xf32>
    ```
  }];
  let arguments = (ins
    StrAttr:$file
  );

  let results = (outs
    AnyTensor:$output
  );

  let hasFolder = 1;
}


#endif // HOM_HOMDIALECT_TD