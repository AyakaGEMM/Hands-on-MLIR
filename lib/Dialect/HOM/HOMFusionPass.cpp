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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "Conversions/Tosa/Passes.h"
#include "HOM/HOMOps.h"
#include "WeightsEngine/WeightsEngine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define PASS_NAME "hom-fusion"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DEF_HOMFUSIONPASS
#include "Dialect/HOM/HOMFusion.pdll.h.inc"
#include "HOM/Passes.h.inc"

namespace {

static LogicalResult checkReshapeRemovableImpl(PatternRewriter &rewriter,
                                               Operation *op0, Operation *op1,
                                               Operation *op2) {
  auto reshapeA = dyn_cast<tosa::ReshapeOp>(op0);
  auto reshapeB = dyn_cast<tosa::ReshapeOp>(op1);
  auto reshapeC = dyn_cast<tosa::ReshapeOp>(op2);

  if (reshapeA && reshapeB && reshapeC) {
    auto inputA = dyn_cast<RankedTensorType>(reshapeA.getOperand().getType());
    auto inputB = dyn_cast<RankedTensorType>(reshapeB.getOperand().getType());
    auto outputC = dyn_cast<RankedTensorType>(reshapeC.getOperand().getType());

    auto newShapeA = reshapeA.getNewShape();
    auto newShapeB = reshapeB.getNewShape();
    auto newShapeC = reshapeC.getNewShape();

    auto oldShapeA = inputA.getShape();
    auto oldShapeB = inputB.getShape();
    auto oldShapeC = outputC.getShape();

    if (oldShapeA.size() != oldShapeB.size() ||
        oldShapeA.size() != newShapeC.size() || oldShapeA.size() < 2) {
      return failure();
    }

    auto M = newShapeA[newShapeA.size() - 2];
    auto N = newShapeB[newShapeB.size() - 1];
    auto K = newShapeA[newShapeA.size() - 1];

    // To-do: Refine MNK check here.
    if (K != newShapeB[newShapeB.size() - 2] ||
        M != oldShapeC[oldShapeC.size() - 2] ||
        N != oldShapeC[oldShapeC.size() - 1]) {
      return failure();
    }

    size_t batchSize = 1;
    for (size_t i = 0; i < oldShapeA.size() - 2; i++) {
      batchSize *= oldShapeA[i];
    }

    auto verifyBatchSizeFn = [&batchSize](const decltype(oldShapeA) &shape) {
      size_t verifyBatchSize = 1;
      for (size_t i = 0; i < shape.size() - 2; i++) {
        verifyBatchSize *= shape[i];
      }
      return verifyBatchSize == batchSize || verifyBatchSize == 1;
    };

    return success(
        verifyBatchSizeFn(oldShapeB) && verifyBatchSizeFn(oldShapeC) &&
        verifyBatchSizeFn(newShapeA) && verifyBatchSizeFn(newShapeB) &&
        verifyBatchSizeFn(newShapeC));
  }
  return failure();
}

static LogicalResult checkMHAQKVReshapeImpl(PatternRewriter &rewriter,
                                            Operation *op0, Operation *op1,
                                            Operation *op2) {
  auto reshapeQ = dyn_cast<tosa::ReshapeOp>(op0);
  auto reshapeK = dyn_cast<tosa::ReshapeOp>(op1);
  auto reshapeV = dyn_cast<tosa::ReshapeOp>(op2);

  if (reshapeQ && reshapeK && reshapeV) {
    auto inputQ = dyn_cast<RankedTensorType>(reshapeQ.getOperand().getType());
    auto inputK = dyn_cast<RankedTensorType>(reshapeK.getOperand().getType());
    auto inputV = dyn_cast<RankedTensorType>(reshapeV.getOperand().getType());

    // newShapes are garenteened to be the same in pdl.
    auto newShape = reshapeQ.getNewShape();

    auto oldShapeQ = inputQ.getShape();
    auto oldShapeK = inputK.getShape();
    auto oldShapeV = inputV.getShape();

    auto checkSameShape = [](const decltype(newShape) &shape0,
                             const decltype(newShape) &shape1) {
      return std::equal(shape0.begin(), shape0.end(), shape1.begin(),
                        shape1.end());
    };

    if (!checkSameShape(oldShapeQ, oldShapeK) ||
        !checkSameShape(oldShapeQ, oldShapeV)) {
      return failure();
    }

    // Check whether qkv reshape from [bs, seq_len, hidden_size] -> [bs,
    // seq_len, head_num, hidden_size].
    auto checkReshapeLegal = [](const decltype(newShape) &oldShape,
                                const decltype(newShape) &newShape) {
      if (oldShape.size() != 3 || newShape.size() != 4) {
        return false;
      }
      // Only need to check the first dimension.
      for (size_t i = 0; i < 2; i++) {
        if (oldShape[i] != newShape[i]) {
          return false;
        }
      }
      return true;
    };

    return success(checkReshapeLegal(oldShapeQ, newShape) &&
                   checkReshapeLegal(oldShapeK, newShape) &&
                   checkReshapeLegal(oldShapeV, newShape));
  }

  return failure();
}

static LogicalResult checkMHAQKVTransposePermImpl(PatternRewriter &rewriter,
                                                  Operation *op0,
                                                  Operation *op1,
                                                  Operation *op2) {
  auto permQ = dyn_cast<tosa::ConstOp>(op0);
  auto permK = dyn_cast<tosa::ConstOp>(op1);
  auto permV = dyn_cast<tosa::ConstOp>(op2);

  if (permQ && permK && permV) {
    auto permQAttr = permQ.getValue();
    auto permKAttr = permK.getValue();
    auto permVAttr = permV.getValue();
    if (permQAttr.getElementType().isIntOrIndex() &&
        permKAttr.getElementType().isIntOrIndex() &&
        permVAttr.getElementType().isIntOrIndex()) {
      auto permQValue = permQAttr.getValues<APInt>();
      auto permKValue = permKAttr.getValues<APInt>();
      auto permVValue = permVAttr.getValues<APInt>();
      SmallVector<uint32_t, 4> refQVPerm = {0, 2, 1, 3};
      SmallVector<uint32_t, 4> refKPerm = {0, 2, 3, 1};

      auto checkPermSame = [](const decltype(permQValue) &permValue,
                              const SmallVector<uint32_t, 4> &refValue) {
        if (permValue.size() != refValue.size()) {
          return false;
        }
        for (size_t i = 0; i < permValue.size(); i++) {
          if (permValue[i].getLimitedValue(32) != refValue[i]) {
            return false;
          }
        }
        return true;
      };
      return success(checkPermSame(permQValue, refQVPerm) &&
                     checkPermSame(permKValue, refKPerm) &&
                     checkPermSame(permVValue, refQVPerm));
    }
  }

  return failure();
}

static LogicalResult
checkTransposeReshapeChangeableImpl(PatternRewriter &rewriter,
                                    Operation *reshape_) {
  auto reshape = dyn_cast<tosa::ReshapeOp>(reshape_);

  auto oldShape = reshape.getInput1().getType().getShape();
  auto newShape = reshape.getNewShape();

  return success(newShape.size() == 3 && oldShape.size() == 2 &&
                 newShape[1] == oldShape[0] && newShape[2] == oldShape[1]);
}

template <class T>
static void createPerm(PatternRewriter &rewriter, const Location &loc,
                       ElementsAttr permAttr, tosa::ConstOp &op) {
  auto permValue = permAttr.getValues<APInt>();
  llvm::SmallVector<T> newPermValue(permValue.size() + 1);
  newPermValue[0] = 0;
  for (size_t i = 0; i < permValue.size(); i++) {
    newPermValue[i + 1] = permValue[i].getLimitedValue(64) + 1;
  }
  auto newPermAttr = DenseElementsAttr::get(
      RankedTensorType::get(ArrayRef<int64_t>{3}, permAttr.getElementType()),
      ArrayRef<T>(newPermValue));
  op = rewriter.create<tosa::ConstOp>(loc, newPermAttr.getType(), newPermAttr);
}

static void changeTransposeReshapeImpl(PatternRewriter &rewriter,
                                       Operation *transpose_, Operation *perm_,
                                       Operation *reshape_) {

  auto loc = transpose_->getLoc();
  auto transpose = dyn_cast<tosa::TransposeOp>(transpose_);
  auto perm = dyn_cast<tosa::ConstOp>(perm_);
  auto reshape = dyn_cast<tosa::ReshapeOp>(reshape_);
  // auto matmul = dyn_cast<hom::MatmulOp>(matmul_);

  auto oldReshapeValue = reshape.getNewShape();
  llvm::SmallVector<int64_t> newShapeValue(oldReshapeValue.size());
  newShapeValue[0] = oldReshapeValue[0];
  newShapeValue[1] = oldReshapeValue[2];
  newShapeValue[2] = oldReshapeValue[1];
  auto newShapeAttr = rewriter.getAttr<DenseI64ArrayAttr>(newShapeValue);

  // auto oldMatmulInput = matmul.getOperand1();
  auto permValue = perm.getValue().getValues<APInt>();
  tosa::ConstOp newPerm;
  if (perm.getValue().getElementType().isInteger(64)) {
    createPerm<int64_t>(rewriter, loc, perm.getValue(), newPerm);
  } else if (perm.getValue().getElementType().isInteger(32)) {
    createPerm<int32_t>(rewriter, loc, perm.getValue(), newPerm);
  } else {
    llvm_unreachable("Not ok");
  }

  auto input = transpose.getInput1();
  auto newReshape = rewriter.create<tosa::ReshapeOp>(loc, input, newShapeAttr);
  auto newTranspose = rewriter.create<tosa::TransposeOp>(
      loc, reshape.getResult().getType(), newReshape.getResult(), newPerm);

  rewriter.replaceOp(reshape, newTranspose);
}

// std::tuple<hom::MatmulOp, hom::BertMhaOp>
static Operation *buildMHAOpImpl(PatternRewriter &rewriter, Operation *reshape_,
                                 Operation *scale_, Value mask, Operation *q_,
                                 Operation *k_, Operation *v_) {
  auto q = dyn_cast<hom::MatmulOp>(q_);
  auto k = dyn_cast<hom::MatmulOp>(k_);
  auto v = dyn_cast<hom::MatmulOp>(v_);
  auto reshape = dyn_cast<tosa::ReshapeOp>(reshape_);
  auto scale = dyn_cast<tosa::ConstOp>(scale_);

  auto qWeight = dyn_cast<tosa::ConstOp>(q.getOperand1().getDefiningOp());
  auto kWeight = dyn_cast<tosa::ConstOp>(k.getOperand1().getDefiningOp());
  auto vWeight = dyn_cast<tosa::ConstOp>(v.getOperand1().getDefiningOp());

  return rewriter.create<hom::BertMhaOp>(
      scale->getLoc(), reshape.getInput1().getType(), reshape.getInput1(), mask,
      scale.getResult());
}

struct HOMFusionPass : impl::HOMFusionPassBase<HOMFusionPass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

LogicalResult HOMFusionPass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);

  populateGeneratedPDLLPatterns(patternList);
  patternList.getPDLPatterns().registerConstraintFunction(
      "checkReshapeRemovable", checkReshapeRemovableImpl);
  patternList.getPDLPatterns().registerConstraintFunction(
      "checkMHAQKVReshape", checkMHAQKVReshapeImpl);
  patternList.getPDLPatterns().registerConstraintFunction(
      "checkMHAQKVTransposePerm", checkMHAQKVTransposePermImpl);
  patternList.getPDLPatterns().registerConstraintFunction(
      "checkTransposeReshapeChangeable", checkTransposeReshapeChangeableImpl);
  patternList.getPDLPatterns().registerRewriteFunction(
      "changeTransposeReshape", changeTransposeReshapeImpl);
  patternList.getPDLPatterns().registerRewriteFunction("buildMHAOp",
                                                       buildMHAOpImpl);
  patterns = std::move(patternList);
  return success();
}

void HOMFusionPass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

} // namespace
} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir
