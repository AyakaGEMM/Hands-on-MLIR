#ifndef HOM_CONVERSIONS_FUNCTION_FUNCTIONCALLUTILS_H_
#define HOM_CONVERSIONS_FUNCTION_FUNCTIONCALLUTILS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace hands_on_mlir {

constexpr llvm::StringRef kAllocF32 = "allocF32";
constexpr llvm::StringRef kAllocDummyTensorF32 = "allocDummyTensorF32";
constexpr llvm::StringRef kAlloc3DMemRefF32 = "alloc3DMemRefF32";
constexpr llvm::StringRef kAlloc1DMemRefNVGPUI32 = "alloc1DMemRefNVGPUI32";
constexpr llvm::StringRef kAlloc3DMemRefNVGPUF32 = "alloc3DMemRefNVGPUF32";
constexpr llvm::StringRef kAlloc1DMemRefNVGPUF16 = "alloc3DMemRefNVGPUF16";
constexpr llvm::StringRef kAlloc3DMemRefNVGPUF16 = "alloc3DMemRefNVGPUF16";
constexpr llvm::StringRef kAllocByMemRefF32 = "allocByMemRefF32";
constexpr llvm::StringRef kAllocConstantF32 = "allocConstantF32";
constexpr llvm::StringRef kAllocConstantF16 = "allocConstantF16";
constexpr llvm::StringRef kAllocConstantNVGPUF32 = "allocConstantNVGPUF32";
constexpr llvm::StringRef kAllocConstantNVGPUF16 = "allocConstantNVGPUF16";
constexpr llvm::StringRef kAllocConstantNVGPUI32 = "allocConstantNVGPUI32";
constexpr llvm::StringRef kArgNum = "_argNum";
constexpr llvm::StringRef kDeallocF32 = "deallocF32";
constexpr llvm::StringRef kDeallocF16 = "deallocF16";
constexpr llvm::StringRef kDeallocI32 = "deallocI32";
constexpr llvm::StringRef kDeallocNVGPUF32 = "deallocNVGPUF32";
constexpr llvm::StringRef kDeallocNVGPUF16 = "deallocNVGPUF16";
constexpr llvm::StringRef kDeallocNVGPUI32 = "deallocNVGPUI32";
constexpr llvm::StringRef kDealloc = "_deallocFn";
constexpr llvm::StringRef kInit = "_initFn";
constexpr llvm::StringRef kMatmulAddF32 = "matmulAddF32";
constexpr llvm::StringRef kMatmulNVGPUF32 = "cutlassMatmulF32";
constexpr llvm::StringRef kGemmNVGPUF32 = "cutlassGemmF32";
constexpr llvm::StringRef kGemmWithVarMeanNVGPUF16 =
    "cutlassGemmWithVarMeanF16";
constexpr llvm::StringRef kGemmNVGPUF16 = "nvteGemmF16";
constexpr llvm::StringRef kCutlassGemmNVGPUF16 = "cutlassGemmF16";
constexpr llvm::StringRef kLayernormNVGPUF32 = "nvteLayernormF32";
constexpr llvm::StringRef kLayernormNVGPUF16 = "nvteLayernormF16";
constexpr llvm::StringRef kBertAttentionNVGPUF32 = "nvteBertAttentionF32";
constexpr llvm::StringRef kBertAttentionNVGPUF16 = "nvteBertAttentionF16";
constexpr llvm::StringRef kCuSeqLenNVGPUI64 = "thrustCuSeqLenI64";
constexpr llvm::StringRef kCuSeqLenNVGPUI32 = "thrustCuSeqLenI32";
constexpr llvm::StringRef kAddNVGPUF32 = "thrustElementwiseAddF32";
constexpr llvm::StringRef kAddNVGPUF16 = "thrustElementwiseAddF16";
constexpr llvm::StringRef kGatherNVGPUF16 = "thrustGatherF16";

func::FuncOp lookupOrCreateFn(ModuleOp moduleOp, StringRef name,
                              ArrayRef<Type> paramTypes,
                              ArrayRef<Type> resultType, bool isPrivate = true);

// FP32
func::FuncOp lookupOrCreateAllocF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAllocDummyTensorF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAlloc3DMemRefF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAlloc3DMemRefNVGPUF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAllocByMemRefF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAllocConstantF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAllocConstantNVGPUF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateArgNumFn(ModuleOp moduleOp, StringRef prefix);
func::FuncOp lookupOrCreateDeallocF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateDeallocNVGPUF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateDeallocFn(ModuleOp moduleOp, StringRef prefix);
func::FuncOp lookupOrCreateInitFn(ModuleOp moduleOp, StringRef prefix);
func::FuncOp lookupOrCreateMatmulAddF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateMatmulF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateMatmulNVGPUF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateGemmNVGPUF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateLayernormNVGPUF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateBertAttentionNVGPUF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAddNVGPUF32Fn(ModuleOp moduleOp);

// FP16
func::FuncOp lookupOrCreateAllocConstantNVGPUF16Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAlloc1DMemRefNVGPUF16Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAlloc3DMemRefNVGPUF16Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateLayernormNVGPUF16Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateBertAttentionNVGPUF16Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateGemmNVGPUF16Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateCutlassGemmNVGPUF16Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateLayernormGemmNVGPUF16Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateGemmWithVarMeanNVGPUF16Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateDeallocF16Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateDeallocNVGPUF16Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateGatherNVGPUF16Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAddNVGPUF16Fn(ModuleOp moduleOp);

// I64
func::FuncOp lookupOrCreateCuSeqLenNVGPUI64Fn(ModuleOp moduleOp);

// I32
func::FuncOp lookupOrCreateAllocConstantNVGPUI32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAlloc1DMemRefNVGPUI32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateCuSeqLenNVGPUI32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateDeallocI32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateDeallocNVGPUI32Fn(ModuleOp moduleOp);

class HOMFuncTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;
  HOMFuncTypeConverter();
};

} // namespace hands_on_mlir
} // namespace mlir

#endif // HOM_CONVERSIONS_FUNCTION_FUNCTIONCALLUTILS_H_
