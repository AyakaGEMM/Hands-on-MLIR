#ifndef HOM_TYPES_BASE
#define HOM_TYPES_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"

def HOM_UInt8 : UI<8>;
def HOM_UInt16 : UI<16>;

def HOM_Int4 : I<4>;
def HOM_Int8 : I<8>;
def HOM_Int16 : I<16>;
def HOM_Int32 : I<32>;
def HOM_Int48 : I<48>;
def HOM_Int64 : I<64>;

def HOM_SignedInt
    : AnyTypeOf<[HOM_Int8, HOM_Int16, HOM_Int32, HOM_Int48, HOM_Int64]>;

def HOM_Bool : I<1>;

// No unsigned unquantized int types.
def HOM_Int : AnyTypeOf<[HOM_Bool, HOM_UInt8, HOM_UInt16, HOM_SignedInt]>;

def HOM_Int32Or64 : AnyTypeOf<[HOM_Int32, HOM_Int64]>;

//===----------------------------------------------------------------------===//
// Floating-point types.
//===----------------------------------------------------------------------===//
def HOM_Float : AnyTypeOf<[F32, F16, BF16]>;

//===----------------------------------------------------------------------===//
// Multi-category types.
//===----------------------------------------------------------------------===//
def HOM_AnyNumber : AnyTypeOf<[HOM_Int, HOM_Float], "number">;

//===----------------------------------------------------------------------===//
// Tensor types
//===----------------------------------------------------------------------===//

def HOM_Int32Tensor : TensorOf<[HOM_Int32]>;
def HOM_Int32Or64Tensor : TensorOf<[HOM_Int32Or64]>;

// Either ranked or unranked tensor of HOM supported element types.
// def HOM_Tensor : TensorOf<[HOM_AnyNumber]>;
def HOM_Tensor : AnyTypeOf<[TensorOf<[HOM_AnyNumber]>]>;

// Must be ranked but no further constraints
def HOM_RankedTensor : RankedTensorOf<[HOM_AnyNumber]>;

def HOM_UnrankedTensor : AnyTypeOf<[UnrankedTensorOf<[HOM_AnyNumber]>]>;

def HOM_Tensor1D
    : AnyTypeOf<[HOM_UnrankedTensor, 1DTensorOf < [HOM_AnyNumber] > ]>;
def HOM_Tensor2D
    : AnyTypeOf<[HOM_UnrankedTensor, 2DTensorOf < [HOM_AnyNumber] > ]>;
def HOM_Tensor3D
    : AnyTypeOf<[HOM_UnrankedTensor, 3DTensorOf < [HOM_AnyNumber] > ]>;
def HOM_Tensor4D
    : AnyTypeOf<[HOM_UnrankedTensor, 4DTensorOf < [HOM_AnyNumber] > ]>;
def HOM_Tensor5D
    : AnyTypeOf<[HOM_UnrankedTensor, TensorRankOf<[HOM_AnyNumber], [5]>]>;

// Any tensor element type allowed in HOM ops.
def HOM_ElementType
    : Type<Or<[HOM_Int.predicate, HOM_Float.predicate]>, "hom.dtype">;

class HOM_TensorOfOrNone<list<Type> allowedTypes, string description = "">
    : AnyTypeOf<[TensorOf<allowedTypes>, NoneType], description>;

def HOM_MemRef
    : AnyTypeOf<[UnrankedMemRefOf<[HOM_AnyNumber]>, MemRefOf<[HOM_AnyNumber]>]>;

def HOM_TensorOrMemRef : AnyTypeOf<[HOM_Tensor, HOM_MemRef]>;

#endif // HOM_TYPES_BASE
