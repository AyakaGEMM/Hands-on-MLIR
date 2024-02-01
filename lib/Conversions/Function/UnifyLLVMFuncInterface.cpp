/* Copyright 2022 OpenXLA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "Conversions/Function/FunctionCallUtils.h"
#include "Conversions/Function/Passes.h"
#include "HOM/HOMOps.h"
#include "WeightsEngine/WeightsEngine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define PASS_NAME "unify-llvm-func-interface"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DEF_UNIFYLLVMFUNCINTERFACEPASS
#include "Conversions/Function/Passes.h.inc"

namespace {

struct UnifyLLVMFuncInterfacePass
    : impl::UnifyLLVMFuncInterfacePassBase<UnifyLLVMFuncInterfacePass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

struct UnifyFuncWithBody : public OpRewritePattern<LLVM::LLVMFuncOp> {
  using OpRewritePattern<LLVM::LLVMFuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getBlocks().empty() ||
        rewrote.find(op.getSymName().str()) != rewrote.end()) {
      return failure();
    }

    auto ctx = op->getContext();
    auto loc = op.getLoc();
    auto llvmPtrTy = rewriter.getType<LLVM::LLVMPointerType>();

    auto unifiedFuncTy = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(ctx), llvmPtrTy, false);

    auto unifiedFunc = rewriter.create<LLVM::LLVMFuncOp>(
        loc, "_hom_ciface_" + op.getSymName().str(), unifiedFuncTy,
        op.getLinkage());

    rewriter.modifyOpInPlace(
        op, [&]() { op.setSymName("_hom_" + op.getSymName().str()); });

    {
      // Rewrite from llvm-project/mlir/lib/ExecutionEngine/ExecutionEngine.cpp
      auto bb = unifiedFunc.addEntryBlock();
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(bb, bb->begin());
      auto loc = unifiedFunc->getLoc();
      auto argsPtr = unifiedFunc.getArgument(0);

      SmallVector<Value> args;

      for (auto [index, arg] : llvm::enumerate(op.getArguments())) {
        auto argIndex = rewriter.create<LLVM::ConstantOp>(
            loc, IntegerType::get(ctx, 64), index);
        auto argPtrPtr = rewriter.create<LLVM::GEPOp>(
            loc, llvmPtrTy, llvmPtrTy, argsPtr, argIndex.getRes());
        auto argPtr =
            rewriter.create<LLVM::LoadOp>(loc, llvmPtrTy, argPtrPtr.getRes());
        auto argTy = arg.getType();
        auto load = rewriter.create<LLVM::LoadOp>(loc, argTy, argPtr);
        args.push_back(load.getRes());
      }

      auto callOldFunc = rewriter.create<LLVM::CallOp>(loc, op, args);

      if (callOldFunc->getNumResults() > 0 &&
          dyn_cast_or_null<LLVM::LLVMVoidType>(
              callOldFunc.getResult().getType()) == nullptr) {
        auto retIndex = rewriter.create<LLVM::ConstantOp>(
            loc, IntegerType::get(ctx, 64), op.getNumArguments());
        auto retPtrPtr = rewriter.create<LLVM::GEPOp>(
            loc, llvmPtrTy, llvmPtrTy, argsPtr, retIndex.getRes());
        Operation *retPtr =
            rewriter.create<LLVM::LoadOp>(loc, llvmPtrTy, retPtrPtr.getRes());
        rewriter.create<LLVM::StoreOp>(loc, callOldFunc.getResult(),
                                       retPtr->getResult(0));
      }

      rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});
    }

    rewrote.insert(unifiedFunc.getSymName().str());
    rewrote.insert(op.getSymName().str());

    return success();
  }

  static std::set<std::string> rewrote;
};

std::set<std::string> UnifyFuncWithBody::rewrote;

LogicalResult UnifyLLVMFuncInterfacePass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);
  patternList.add<UnifyFuncWithBody>(ctx);
  patterns = std::move(patternList);
  UnifyFuncWithBody::rewrote.clear();
  return success();
}

void UnifyLLVMFuncInterfacePass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

} // namespace
} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir
