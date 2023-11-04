#include "Conversions/Function/FunctionCallUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace mlir {
namespace hands_on_mlir {
namespace hom {

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

func::FuncOp lookupOrCreateMatmulAddF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kMatmulAddF32,
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float32Type::get(ctx), 0),
                           UnrankedMemRefType::get(Float32Type::get(ctx), 0)},
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

func::FuncOp lookupOrCreateAllocByMemRefF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kAllocByMemRefF32,
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0)},
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0)});
}

func::FuncOp lookupOrCreateDeallocF32Fn(ModuleOp moduleOp) {
  auto ctx = moduleOp->getContext();
  return lookupOrCreateFn(moduleOp, kDeallocF32,
                          {UnrankedMemRefType::get(Float32Type::get(ctx), 0)},
                          {});
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

} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir