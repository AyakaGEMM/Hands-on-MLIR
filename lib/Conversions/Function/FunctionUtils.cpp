#include "Conversions/Function/FunctionUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Twine.h"

namespace mlir {
namespace hands_on_mlir {

/// Generic print function lookupOrCreate helper.
func::FuncOp lookupOrCreateFn(ModuleOp moduleOp, StringRef name,
                              ArrayRef<Type> paramTypes,
                              ArrayRef<Type> resultType, bool isPrivate) {
  auto ctx = moduleOp->getContext();
  auto func = moduleOp.lookupSymbol<func::FuncOp>(name);
  if (func) {
    return func;
  }
  OpBuilder b(moduleOp.getBodyRegion());
  auto createFn = b.create<func::FuncOp>(
      moduleOp->getLoc(), name,
      FunctionType::get(moduleOp->getContext(), paramTypes, resultType));
  if (isPrivate) {
    createFn.setSymVisibilityAttr(StringAttr::get(ctx, "private"));
  }
  return createFn;
}

func::FuncOp lookupOrCreateAllocConstantF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp.getContext();
  return lookupOrCreateFn(moduleOp, kAllocConstantF32,
                          {IntegerType::get(ctx, 32)},
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0)});
}

func::FuncOp lookupOrCreateAllocDummyTensorF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp.getContext();
  return lookupOrCreateFn(moduleOp, kAllocDummyTensorF32, {},
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0)});
}

func::FuncOp lookupOrCreateAllocConstantNVGPUF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp.getContext();
  return lookupOrCreateFn(moduleOp, kAllocConstantNVGPUF32,
                          {IntegerType::get(ctx, 32)},
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0)});
}

func::FuncOp lookupOrCreateAllocConstantNVGPUF16Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp.getContext();
  return lookupOrCreateFn(moduleOp, kAllocConstantNVGPUF16,
                          {IntegerType::get(ctx, 32)},
                          {UnrankedMemRefType::get(Float16Type::get(ctx), 0)});
}

func::FuncOp lookupOrCreateMatmulAddF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kMatmulAddF32,
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float32Type::get(ctx), 0)},
                          {});
}

func::FuncOp lookupOrCreateMatmulNVGPUF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kMatmulAddF32,
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float32Type::get(ctx), 0)},
                          {});
}

func::FuncOp lookupOrCreateGemmNVGPUF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kGemmNVGPUF32,
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           IntegerType::get(ctx, 1),
                           UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           IntegerType::get(ctx, 1),
                           UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           IntegerType::get(ctx, 64), Float32Type::get(ctx),
                           Float32Type::get(ctx)},
                          {});
}

func::FuncOp lookupOrCreateGemmNVGPUF16Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kGemmNVGPUF16,
                          {UnrankedMemRefType::get(Float16Type::get(ctx), 0),
                           IntegerType::get(ctx, 1),
                           UnrankedMemRefType::get(Float16Type::get(ctx), 0),
                           IntegerType::get(ctx, 1),
                           UnrankedMemRefType::get(Float16Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float16Type::get(ctx), 0),
                           IntegerType::get(ctx, 64), Float32Type::get(ctx),
                           Float32Type::get(ctx)},
                          {});
}

func::FuncOp lookupOrCreateLayernormNVGPUF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kLayernormNVGPUF32,
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           Float32Type::get(ctx)},
                          {});
}

func::FuncOp lookupOrCreateLayernormNVGPUF16Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kLayernormNVGPUF16,
                          {UnrankedMemRefType::get(Float16Type::get(ctx), 0),
                           Float32Type::get(ctx)},
                          {});
}

func::FuncOp lookupOrCreateGatherNVGPUF16Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(
      moduleOp, kGatherNVGPUF16,
      {UnrankedMemRefType::get(IntegerType::get(ctx, 64), 0),
       UnrankedMemRefType::get(Float16Type::get(ctx), 0),
       UnrankedMemRefType::get(Float16Type::get(ctx), 0)},
      {});
}

func::FuncOp lookupOrCreateAddNVGPUF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kAddNVGPUF32,
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float32Type::get(ctx), 0)},
                          {});
}

func::FuncOp lookupOrCreateAddNVGPUF16Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kAddNVGPUF16,
                          {UnrankedMemRefType::get(Float16Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float16Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float16Type::get(ctx), 0)},
                          {});
}

func::FuncOp lookupOrCreateBertAttentionNVGPUF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(
      moduleOp, kBertAttentionNVGPUF32,
      {UnrankedMemRefType::get(Float32Type::get(ctx), 0),
       UnrankedMemRefType::get(IntegerType::get(ctx, 32), 0),
       UnrankedMemRefType::get(Float32Type::get(ctx), 0), Float32Type::get(ctx),
       IntegerType::get(ctx, 64)},
      {});
}

func::FuncOp lookupOrCreateBertAttentionNVGPUF16Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(
      moduleOp, kBertAttentionNVGPUF16,
      {UnrankedMemRefType::get(Float16Type::get(ctx), 0),
       UnrankedMemRefType::get(IntegerType::get(ctx, 32), 0),
       UnrankedMemRefType::get(Float16Type::get(ctx), 0), Float32Type::get(ctx),
       IntegerType::get(ctx, 64)},
      {});
}

func::FuncOp lookupOrCreateAllocF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kAllocF32, {IntegerType::get(ctx, 32)},
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0)});
}

func::FuncOp lookupOrCreateAlloc3DMemRefF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kAlloc3DMemRefF32,
                          {IntegerType::get(ctx, 32), IntegerType::get(ctx, 32),
                           IntegerType::get(ctx, 32)},
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0)});
}

func::FuncOp lookupOrCreateAlloc3DMemRefNVGPUF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kAlloc3DMemRefNVGPUF32,
                          {IntegerType::get(ctx, 32), IntegerType::get(ctx, 32),
                           IntegerType::get(ctx, 32)},
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0)});
}

func::FuncOp lookupOrCreateAlloc1DMemRefNVGPUI32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(
      moduleOp, kAlloc1DMemRefNVGPUI32, {IntegerType::get(ctx, 32)},
      {UnrankedMemRefType::get(IntegerType::get(ctx, 32), 0)});
}

func::FuncOp lookupOrCreateAllocConstantNVGPUI32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp.getContext();
  return lookupOrCreateFn(
      moduleOp, kAllocConstantNVGPUF16, {IntegerType::get(ctx, 32)},
      {UnrankedMemRefType::get(IntegerType::get(ctx, 32), 0)});
}

func::FuncOp lookupOrCreateCuSeqLenNVGPUI64Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(
      moduleOp, kCuSeqLenNVGPUI64,
      {UnrankedMemRefType::get(IntegerType::get(ctx, 64), 0),
       UnrankedMemRefType::get(IntegerType::get(ctx, 32), 0)},
      {});
}

func::FuncOp lookupOrCreateCuSeqLenNVGPUI32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(
      moduleOp, kCuSeqLenNVGPUI32,
      {UnrankedMemRefType::get(IntegerType::get(ctx, 32), 0),
       UnrankedMemRefType::get(IntegerType::get(ctx, 32), 0)},
      {});
}

func::FuncOp lookupOrCreateAlloc3DMemRefNVGPUF16Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kAlloc3DMemRefNVGPUF16,
                          {IntegerType::get(ctx, 32), IntegerType::get(ctx, 32),
                           IntegerType::get(ctx, 32)},
                          {UnrankedMemRefType::get(Float16Type::get(ctx), 0)});
}

func::FuncOp lookupOrCreateAllocByMemRefF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kAllocByMemRefF32,
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0)},
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0)});
}

func::FuncOp lookupOrCreateArgNumFn(ModuleOp moduleOp, StringRef prefix) {
  auto ctx = moduleOp.getContext();
  SmallVector<char> name;
  return lookupOrCreateFn(moduleOp, (prefix + kArgNum).toStringRef(name), {},
                          IntegerType::get(ctx, 32));
}

func::FuncOp lookupOrCreateDeallocF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kDeallocF32,
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0)},
                          {});
}

func::FuncOp lookupOrCreateDeallocF16Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kDeallocF16,
                          {UnrankedMemRefType::get(Float16Type::get(ctx), 0)},
                          {});
}

func::FuncOp lookupOrCreateDeallocI32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(
      moduleOp, kDeallocI32,
      {UnrankedMemRefType::get(IntegerType::get(ctx, 32), 0)}, {});
}

func::FuncOp lookupOrCreateDeallocNVGPUF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kDeallocNVGPUF32,
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0)},
                          {});
}

func::FuncOp lookupOrCreateDeallocNVGPUF16Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kDeallocNVGPUF16,
                          {UnrankedMemRefType::get(Float16Type::get(ctx), 0)},
                          {});
}

func::FuncOp lookupOrCreateDeallocNVGPUI32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(
      moduleOp, kDeallocNVGPUI32,
      {UnrankedMemRefType::get(IntegerType::get(ctx, 32), 0)}, {});
}

func::FuncOp lookupOrCreateInitFn(ModuleOp moduleOp, StringRef prefix) {
  SmallVector<char> name;
  return lookupOrCreateFn(moduleOp, (prefix + kInit).toStringRef(name), {}, {});
}

func::FuncOp lookupOrCreateDeallocFn(ModuleOp moduleOp, StringRef prefix) {
  SmallVector<char> name;
  return lookupOrCreateFn(moduleOp, (prefix + kDealloc).toStringRef(name), {},
                          {});
}

HOMFuncTypeConverter::HOMFuncTypeConverter() {
  this->addConversion([](Type type) { return type; });
  this->addConversion([](RankedTensorType type) {
    return UnrankedMemRefType::get(type.getElementType(), 0);
  });
  this->addConversion([](UnrankedTensorType type) -> Type {
    return UnrankedMemRefType::get(type.getElementType(), 0);
  });
}

} // namespace hands_on_mlir
} // namespace mlir
