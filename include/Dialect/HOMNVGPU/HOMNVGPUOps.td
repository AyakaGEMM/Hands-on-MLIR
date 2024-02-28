#ifndef HOM_HOMNVGPUDIALECT_TD
#define HOM_HOMNVGPUDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "HOM/HOMTypesBase.td" // Just reuses the hom types.
include "mlir/Interfaces/InferTypeOpInterface.td"

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
    : Op<HOMNVGPU_Dialect, mnemonic, !listconcat(traits, [])>;

def HOMNVGPU_PrintOp : HOMNVGPU_Op<"print", []> {
  let arguments = (ins F64Tensor : $input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def HOMNVGPU_MatmulOp : HOMNVGPU_Op<"matmul", []> {
  let arguments = (ins HOM_RankedTensor
                   : $operand0, HOM_RankedTensor
                   : $operand1, HOM_Tensor
                   : $operand2, F32Attr
                   : $alpha, F32Attr
                   : $beta, I32Attr
                   : $act, DefaultValuedAttr<BoolAttr, "false"> : $transa, DefaultValuedAttr<BoolAttr, "false"> : $transb);

  let results = (outs HOM_RankedTensor : $output);
}

def HOMNVGPU_MatmulWithVarMeanOp
    : HOMNVGPU_Op<"matmul_with_var_mean",
                  [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins HOM_RankedTensor
                   : $operand0, HOM_RankedTensor
                   : $operand1, HOM_Tensor
                   : $operand2, F32Attr
                   : $alpha, F32Attr
                   : $beta, I32Attr
                   : $act, F32Attr
                   : $eps);

  let results = (outs HOM_RankedTensor
                 : $output, HOM_RankedTensor
                 : $var, HOM_RankedTensor
                 : $mean);
}

def HOMNVGPU_LayernormMatmul : HOMNVGPU_Op<"ln_matmul", []> {
  let arguments = (ins HOM_RankedTensor
                   : $operand0, HOM_RankedTensor
                   : $operand1, HOM_Tensor
                   : $operand2, HOM_RankedTensor
                   : $var, HOM_RankedTensor
                   : $mean, F32Attr
                   : $alpha, F32Attr
                   : $beta, I32Attr
                   : $act);

  let results = (outs HOM_RankedTensor : $output);
}

def HOMNVGPU_LayernormOp : HOMNVGPU_Op<"layernorm", []> {
  let summary = "Layernorm operator";

  let description = [{Do layernorm.}];

  let arguments = (ins HOM_Tensor : $input, F32Attr : $eps, I32Attr : $axis);

  let results = (outs HOM_Tensor : $output);
}

def HOMNVGPU_BertMhaOp : HOMNVGPU_Op<"bert_mha", []> {
  let summary = "Bert's MHA operator";

  let description = [{Do Bert MHA.}];

  let arguments = (ins HOM_RankedTensor
                   : $qkv, HOM_RankedTensor
                   : $mask, F32Attr
                   : $scale, I64Attr
                   : $head_num);

  let results = (outs HOM_RankedTensor : $output);
}

def HOMNVGPU_CuSeqLenOp : HOMNVGPU_Op<"cu_seqlen", []> {
  let summary = "Bert's MHA operator";

  let description = [{Do Bert MHA.}];

  let arguments = (ins HOM_RankedTensor : $input);

  let results = (outs HOM_RankedTensor : $output);
}

def HOMNVGPU_AddOp : HOMNVGPU_Op<"add", [Commutative]> {
  let summary = "Elementwise addition operator";

  let description =
      [{Elementwise addition of input1 and
            input2.Axis of size 1 will be broadcast as necessary.}];

  let arguments = (ins HOM_Tensor : $input1, HOM_Tensor : $input2);

  let results = (outs HOM_Tensor : $output);
}

def HOMNVGPU_GatherOp : HOMNVGPU_Op<"gather", []> {
  let arguments = (ins HOM_RankedTensor : $indices, HOM_RankedTensor : $value);

  let results = (outs HOM_RankedTensor : $output);
}

#endif // HOM_HOMNVGPUDIALECT_TD
