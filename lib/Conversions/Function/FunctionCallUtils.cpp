#include "Conversions/Function/FunctionCallUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hands_on_mlir {
namespace hom {

/// Generic print function lookupOrCreate helper.
func::FuncOp lookupOrCreateFn(ModuleOp moduleOp, StringRef name,
                              ArrayRef<Type> paramTypes,
                              ArrayRef<Type> resultType) {
  auto func = moduleOp.lookupSymbol<func::FuncOp>(name);
  if (func)
    return func;
  OpBuilder b(moduleOp.getBodyRegion());
  return b.create<func::FuncOp>(
      moduleOp->getLoc(), name,
      FunctionType::get(moduleOp->getContext(), paramTypes, resultType));
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

} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir