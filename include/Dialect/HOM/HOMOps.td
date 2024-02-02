#ifndef HOM_HOMDIALECT_TD
#define HOM_HOMDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "HOM/HOMTypesBase.td"

//===----------------------------------------------------------------------===//
// HOM Dialect Definition.
//===----------------------------------------------------------------------===//

def HOM_Dialect : Dialect {
  let name = "hom";
  let summary = "The Hands on MLIR Main Dialect.";
  let description = [{ The `HOM` dialect is for better op management. }];
  let cppNamespace = "::mlir::hands_on_mlir::hom";
}

//===----------------------------------------------------------------------===//
// Base HOM Operation Definition.
//===----------------------------------------------------------------------===//

class HOM_Op<string mnemonic, list<Trait> traits = []>
    : Op<HOM_Dialect, mnemonic, traits>;

def HOM_PrintOp : HOM_Op<"print", []> {
  let arguments = (ins F64Tensor : $input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def HOM_ConstantOp : HOM_Op<"constant", [ConstantLike, Pure]> {
  let summary = "Constant operation";
  let description = [{
    Produces an `output` tensor from a constant `value`.

    Example:
    ```mlir % output =
        hom.constant dense<[[0.0, 1.0], [2.0, 3.0]]>: tensor<2x2xf32>
    ```
  }];

  let arguments = (ins
                       // StrAttr:$file,
                       I64Attr
                   : $idx);

  let results = (outs HOM_Tensor : $output);

  let hasFolder = 1;
}

def HOM_TransposeOp
    : HOM_Op<"transpose", [Pure, SameOperandsAndResultElementType]> {
  let summary = "Transpose operation";
  let description = [{
    Permutes the dimensions of `operand` tensor using `permutation` and produces
        a `result` tensor.

    Example:
    ```mlir % 0 = hom.transpose % arg0,
    dims = [2, 1, 0]: (tensor<1x2x3xi32>)->tensor<3x2x1xi32>
    ```
  }];

  let arguments = (ins HOM_Tensor
                   : $operand, HOM_Int32Or64Tensor
                   : $permutation);

  let results = (outs HOM_Tensor : $output);
}

def HOM_MatmulOp : HOM_Op<"matmul", [Pure]> {
  let summary = "Transpose operation";
  let description = [{
    Permutes the dimensions of `operand` tensor using `permutation` and produces
        a `result` tensor.

    Example:
    ```mlir % 0 = hom.transpose % arg0,
    dims = [2, 1, 0]: (tensor<1x2x3xi32>)->tensor<3x2x1xi32>
    ```
  }];

  let arguments = (ins HOM_Tensor3D : $operand0, HOM_Tensor3D : $operand1);

  let results = (outs HOM_Tensor3D : $output);
}

def HOM_MatmulAddOp : HOM_Op<"matmul_add", [Pure]> {
  let summary = "Transpose operation";
  let description = [{
    Permutes the dimensions of `operand` tensor using `permutation` and produces
        a `result` tensor.

    Example:
    ```mlir % 0 = hom.transpose % arg0,
    dims = [2, 1, 0]: (tensor<1x2x3xi32>)->tensor<3x2x1xi32>
    ```
  }];

  let arguments = (ins HOM_Tensor3D
                   : $operand0, HOM_Tensor3D
                   : $operand1, HOM_Tensor
                   : $operand2);

  let results = (outs HOM_Tensor3D : $output);
}

def HOM_AddOp : HOM_Op<"add", [Commutative, SameOperandsAndResultElementType]> {
  let summary = "Elementwise addition operator";

  let description =
      [{Elementwise addition of input1 and
            input2.Axis of size 1 will be broadcast as necessary.}];

  let arguments = (ins HOM_Tensor : $input1, HOM_Tensor : $input2);

  let results = (outs HOM_Tensor : $output);
}

def HOM_SoftmaxOp
    : HOM_Op<"softmax", [Commutative, SameOperandsAndResultElementType]> {
  let summary = "Softmax operator";

  let description = [{Do softmax.}];

  let arguments = (ins HOM_Tensor : $input, I32Attr : $axis);

  let results = (outs HOM_Tensor : $output);
}

def HOM_LayernormOp
    : HOM_Op<"layernorm", [Commutative, SameOperandsAndResultElementType]> {
  let summary = "Layernorm operator";

  let description = [{Do layernorm.}];

  let arguments = (ins HOM_Tensor : $input, F32Attr : $eps, I32Attr : $axis);

  let results = (outs HOM_Tensor : $output);
}

def HOM_GeluOp
    : HOM_Op<"gelu", [Commutative, SameOperandsAndResultElementType]> {
  let summary = "Gelu operator";

  let description = [{Do Gelu.}];

  let arguments = (ins HOM_Tensor : $input);

  let results = (outs HOM_Tensor : $output);
}

#endif // HOM_HOMDIALECT_TD
