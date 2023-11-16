//===- FunctionCallUtils.h - Utilities for C function calls -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helper functions to call common simple C functions in
// LLVMIR (e.g. among others to support printing and debugging).
//
//===----------------------------------------------------------------------===//

#ifndef HOM_CONVERSIONS_FUNCTION_FUNCTIONCALLUTILS_H_
#define HOM_CONVERSIONS_FUNCTION_FUNCTIONCALLUTILS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace hands_on_mlir {
namespace hom {

constexpr llvm::StringRef kAllocF32 = "allocF32";
constexpr llvm::StringRef kAlloc3DMemRefF32 = "alloc3DMemRefF32";
constexpr llvm::StringRef kAllocByMemRefF32 = "allocByMemRefF32";
constexpr llvm::StringRef kAllocConstantF32 = "allocConstantF32";
constexpr llvm::StringRef kArgNum = "_argNum";
constexpr llvm::StringRef kDeallocF32 = "deallocF32";
constexpr llvm::StringRef kDealloc = "_deallocFn";
constexpr llvm::StringRef kInit = "_initFn";
constexpr llvm::StringRef kMatmulAddF32 = "matmulAddF32";

func::FuncOp lookupOrCreateFn(ModuleOp moduleOp, StringRef name,
                              ArrayRef<Type> paramTypes,
                              ArrayRef<Type> resultType, bool isPrivate = true);
func::FuncOp lookupOrCreateAllocF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAlloc3DMemRefF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAllocByMemRefF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateAllocConstantF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateArgNumFn(ModuleOp moduleOp, StringRef prefix);
func::FuncOp lookupOrCreateDeallocF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateDeallocFn(ModuleOp moduleOp, StringRef prefix);
func::FuncOp lookupOrCreateInitFn(ModuleOp moduleOp, StringRef prefix);
func::FuncOp lookupOrCreateMatmulAddF32Fn(ModuleOp moduleOp);
func::FuncOp lookupOrCreateMatmulF32Fn(ModuleOp moduleOp);

} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir

#endif // HOM_CONVERSIONS_FUNCTION_FUNCTIONCALLUTILS_H_
