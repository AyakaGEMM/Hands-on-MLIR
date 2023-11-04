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
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "Conversions/Function/FunctionCallUtils.h"
#include "Conversions/Function/Passes.h"
#include "HOM/HOMOps.h"
#include "WeightsEngine/WeightsEngine.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

#define PASS_NAME "extract-init-func"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DEF_EXTRACTINITFUNCPASS
#include "Conversions/Function/Passes.h.inc"

namespace {

struct ExtractInitFuncPass
    : impl::ExtractInitFuncPassBase<ExtractInitFuncPass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

struct ExtractPattern : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getBlocks().empty() ||
        handled.find(op.getSymName().str()) != handled.end()) {
      return failure();
    }

    auto ctx = op.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto initFn = lookupOrCreateInitFn(moduleOp, op.getSymName());
    auto deallocFn = lookupOrCreateDeallocFn(moduleOp, op.getSymName());

    initFn.addEntryBlock();
    deallocFn.addEntryBlock();

    OpBuilder builder(initFn.getFunctionBody()),
        builder1(deallocFn.getFunctionBody());
    builder1.create<func::ReturnOp>(deallocFn->getLoc());

    SmallVector<Operation *> op2remove, alloc2remove;
    SmallVector<Type> initAllocTypes, fnArgTypes(op.getArgumentTypes());
    SmallVector<Value> initAllocValues;

    op->walk([&](func::CallOp callOp) {
      if (callOp.getCallee().equals("allocConstantF32")) {
        alloc2remove.emplace_back(callOp);
        auto idxOp = dyn_cast<arith::ConstantIntOp>(
            callOp->getOperand(0).getDefiningOp());
        if (!idxOp) {
          llvm_unreachable("Not Good.");
        }
        for (auto user : callOp->getUsers()) {
          if (auto dealloc = dyn_cast<func::CallOp>(user)) {
            if (dealloc.getCallee().contains("dealloc")) {
              user->dump();
              op2remove.emplace_back(dealloc);
            }
          }
        }
        auto initIdxOp = builder.create<arith::ConstantIntOp>(
            initFn.getLoc(), idxOp.value(), 32);
        auto initCallOp = builder.create<func::CallOp>(
            initFn.getLoc(), callOp.getCallee(), callOp->getResultTypes(),
            ValueRange{initIdxOp.getResult()});
        initAllocValues.emplace_back(initCallOp->getResult(0));
        initAllocTypes.emplace_back(initCallOp->getResult(0).getType());
        fnArgTypes.emplace_back(initCallOp->getResult(0).getType());
      }
    });

    builder.create<func::ReturnOp>(initFn->getLoc(), initAllocValues);
    auto opFnTy = FunctionType::get(ctx, fnArgTypes, op->getResultTypes());
    auto initFnTy = FunctionType::get(ctx, {}, initAllocTypes);
    auto deallocFnTy = FunctionType::get(ctx, initAllocTypes, {});
    initFn.setFunctionType(initFnTy);
    deallocFn.setFunctionType(deallocFnTy);

    int idx = 0, offset = op->getNumOperands();

    std::cout << deallocFn.getNumArguments() << std::endl;

    auto aa = deallocFn.getArguments().size();

    std::cout << aa << std::endl;

    rewriter.updateRootInPlace(op, [&]() { op.setFunctionType(opFnTy); });

    for (auto type : initAllocTypes) {
      auto memrefTy = dyn_cast<UnrankedMemRefType>(type);
      if (!memrefTy) {
        llvm_unreachable("Not Good.");
      }
      if (memrefTy.getElementType().isF32()) {
        auto internalDeallocFn = lookupOrCreateDeallocF32Fn(moduleOp);
        // builder1.create<func::CallOp>(deallocFn.getLoc(), internalDeallocFn,
        //                               ValueRange{deallocFn.getArgument(idx)});
        std::cout << idx << std::endl;
      } else {
        llvm_unreachable("Not Good.");
      }
      idx++;
    }

    for (auto &op : alloc2remove) {
      while (op->getUses().empty()) {
        op->getUses().begin()->set(op->getOperand(offset++));
      }
      // rewriter.eraseOp(op);
    }

    for (auto &op : op2remove) {
      rewriter.eraseOp(op);
    }

    handled.insert(initFn.getSymName().str());
    handled.insert(deallocFn.getSymName().str());
    handled.insert(op.getSymName().str());

    return success();
  }
  static std::set<std::string> handled;
};

std::set<std::string> ExtractPattern::handled;

LogicalResult ExtractInitFuncPass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);
  patternList.add<ExtractPattern>(ctx);
  patterns = std::move(patternList);
  ExtractPattern::handled.clear();
  return success();
}

void ExtractInitFuncPass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

} // namespace
} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir
