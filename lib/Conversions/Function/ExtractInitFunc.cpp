#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "Conversions/Function/FunctionUtils.h"
#include "Conversions/Function/Passes.h"
#include "HOM/HOMOps.h"
#include "WeightsEngine/WeightsEngine.h"
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
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

    OpBuilder initFnBuilder(initFn.getFunctionBody()),
        deallocFnbuilder(deallocFn.getFunctionBody());

    SmallVector<Operation *> dealloc2remove, alloc2remove;
    SmallVector<std::pair<Type, int32_t>> initAllocTypes;
    SmallVector<Value> initAllocValues;

    op->walk([&](func::CallOp callOp) {
      if (callOp.getCallee().contains("alloc")) {
        auto res = callOp->getResult(0);
        if (res.getUsers().empty()) {
          rewriter.eraseOp(callOp);
          return;
        }
      }

      if (callOp.getCallee().equals(kAllocConstantF32) ||
          callOp.getCallee().equals(kAllocConstantNVGPUF32) ||
          callOp.getCallee().equals(kAllocConstantNVGPUF16)) {
        alloc2remove.emplace_back(callOp);
        auto idxOp = dyn_cast<arith::ConstantIntOp>(
            callOp->getOperand(0).getDefiningOp());
        if (!idxOp) {
          llvm_unreachable("Not Good.");
        }
        for (auto user : callOp->getUsers()) {
          if (auto dealloc = dyn_cast<func::CallOp>(user)) {
            if (dealloc.getCallee().contains("dealloc")) {
              dealloc2remove.emplace_back(dealloc);
            }
          }
        }
        auto initIdxOp = initFnBuilder.create<arith::ConstantIntOp>(
            initFn.getLoc(), idxOp.value(), 32);
        auto initCallOp = initFnBuilder.create<func::CallOp>(
            initFn.getLoc(), callOp.getCallee(), callOp->getResultTypes(),
            ValueRange{initIdxOp.getResult()});
        initAllocValues.emplace_back(initCallOp->getResult(0));
        initAllocTypes.emplace_back(
            std::make_pair(initCallOp->getResult(0).getType(),
                           callOp.getCallee().contains("NVGPU")));
      } else if (callOp.getCallee().equals(kAlloc3DMemRefF32) ||
                 callOp.getCallee().equals(kAlloc3DMemRefNVGPUF32) ||
                 callOp.getCallee().equals(kAlloc3DMemRefNVGPUF16)) {
        alloc2remove.emplace_back(callOp);

        SmallVector<Value> operands;
        for (int i = 0; i < 3; i++) {
          auto idxOp = dyn_cast<arith::ConstantIntOp>(
              callOp->getOperand(i).getDefiningOp());
          if (!idxOp) {
            llvm_unreachable("Not Good.");
          }
          auto initIdxOp = initFnBuilder.create<arith::ConstantIntOp>(
              initFn.getLoc(), idxOp.value(), 32);
          operands.emplace_back(initIdxOp);
        }

        auto initCallOp = initFnBuilder.create<func::CallOp>(
            initFn.getLoc(), callOp.getCallee(), callOp->getResultTypes(),
            operands);
        initAllocValues.emplace_back(initCallOp->getResult(0));
        initAllocTypes.emplace_back(
            std::make_pair(initCallOp->getResult(0).getType(),
                           callOp.getCallee().contains("NVGPU")));

        for (auto user : callOp->getUsers()) {
          if (auto dealloc = dyn_cast<func::CallOp>(user)) {
            if (dealloc.getCallee().contains("dealloc")) {
              dealloc2remove.emplace_back(dealloc);
            }
          }
        }
      } else if (callOp.getCallee().equals(kAllocDummyTensorF32)) {
        if (callOp->getUsers().empty()) {
          rewriter.eraseOp(callOp);
        } else {
          llvm_unreachable("Dummy tensor should not be used in anyway.");
        }
      } else if (callOp.getCallee().equals(kAlloc1DMemRefNVGPUI32)) {
        alloc2remove.emplace_back(callOp);

        SmallVector<Value> operands;
        auto idxOp = dyn_cast<arith::ConstantIntOp>(
            callOp->getOperand(0).getDefiningOp());
        if (!idxOp) {
          llvm_unreachable("Not Good.");
        }
        auto initIdxOp = initFnBuilder.create<arith::ConstantIntOp>(
            initFn.getLoc(), idxOp.value(), 32);
        operands.emplace_back(initIdxOp);

        auto initCallOp = initFnBuilder.create<func::CallOp>(
            initFn.getLoc(), callOp.getCallee(), callOp->getResultTypes(),
            operands);
        initAllocValues.emplace_back(initCallOp->getResult(0));
        initAllocTypes.emplace_back(
            std::make_pair(initCallOp->getResult(0).getType(),
                           callOp.getCallee().contains("NVGPU")));

        for (auto user : callOp->getUsers()) {
          if (auto dealloc = dyn_cast<func::CallOp>(user)) {
            if (dealloc.getCallee().contains("dealloc")) {
              dealloc2remove.emplace_back(dealloc);
            }
          }
        }
      }
    });

    initFnBuilder.create<func::ReturnOp>(initFn->getLoc(), initAllocValues);

    int idx = 0;
    /// offset = op.getNumArguments();
    auto unknownLoc = UnknownLoc::get(ctx);
    SmallVector<unsigned int> argIndices;
    SmallVector<Location> argLoc;
    SmallVector<DictionaryAttr> argAttr;
    SmallVector<Type> argType;
    for (size_t i = 0; i < initAllocTypes.size(); i++) {
      argIndices.emplace_back(0);
      argLoc.emplace_back(unknownLoc);
      argAttr.emplace_back(DictionaryAttr::get(ctx));
      argType.emplace_back(initAllocTypes[i].first);
    }

    auto initFnTy = FunctionType::get(ctx, {}, argType);
    initFn.setFunctionType(initFnTy);

    rewriter.modifyOpInPlace(op, [&]() {
      op.insertArguments(argIndices, argType, argAttr, argLoc);
    });

    deallocFn.insertArguments(argIndices, argType, argAttr, argLoc);

    for (auto type : initAllocTypes) {
      auto memrefTy = dyn_cast<UnrankedMemRefType>(type.first);
      if (!memrefTy) {
        llvm_unreachable("Not Good.");
      }

      if (memrefTy.getElementType().isF32()) {
        auto internalDeallocFn = type.second
                                     ? lookupOrCreateDeallocNVGPUF32Fn(moduleOp)
                                     : lookupOrCreateDeallocF32Fn(moduleOp);
        deallocFnbuilder.create<func::CallOp>(
            deallocFn.getLoc(), internalDeallocFn,
            ValueRange{deallocFn.getArgument(idx)});
      } else if (memrefTy.getElementType().isF16()) {
        auto internalDeallocFn = type.second
                                     ? lookupOrCreateDeallocNVGPUF16Fn(moduleOp)
                                     : lookupOrCreateDeallocF16Fn(moduleOp);
        deallocFnbuilder.create<func::CallOp>(
            deallocFn.getLoc(), internalDeallocFn,
            ValueRange{deallocFn.getArgument(idx)});
      } else if (memrefTy.getElementType().isInteger(32)) {
        auto internalDeallocFn = type.second
                                     ? lookupOrCreateDeallocNVGPUI32Fn(moduleOp)
                                     : lookupOrCreateDeallocI32Fn(moduleOp);
        deallocFnbuilder.create<func::CallOp>(
            deallocFn.getLoc(), internalDeallocFn,
            ValueRange{deallocFn.getArgument(idx)});
      } else {
        llvm_unreachable("Not Good.");
      }
      idx++;
    }

    deallocFnbuilder.create<func::ReturnOp>(deallocFn->getLoc());

    idx = 0;
    for (auto &allocOp : alloc2remove) {
      while (!allocOp->getUses().empty()) {
        allocOp->getUses().begin()->set(op.getArgument(idx));
      }
      idx++;
      rewriter.eraseOp(allocOp);
    }

    for (auto &op : dealloc2remove) {
      rewriter.eraseOp(op);
    }

    auto argNumFn = lookupOrCreateArgNumFn(moduleOp, op.getSymName());
    argNumFn.addEntryBlock();

    OpBuilder numFnBuilder(argNumFn.getBody());
    auto argActuallNum = numFnBuilder.create<arith::ConstantIntOp>(
        argNumFn->getLoc(), initAllocTypes.size(), 32);
    numFnBuilder.create<func::ReturnOp>(argNumFn->getLoc(),
                                        ValueRange{argActuallNum.getResult()});

    handled.insert(initFn.getSymName().str());
    handled.insert(deallocFn.getSymName().str());
    handled.insert(op.getSymName().str());
    handled.insert(argNumFn.getSymName().str());

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
} // namespace hands_on_mlir
} // namespace mlir
