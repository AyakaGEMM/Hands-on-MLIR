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
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "Conversions/Tosa/Passes.h"
#include "ExecutionEngine/HandsOnNVGPURunnerUtils.h"
#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "HOMNVGPU/HOMNVGPUOps.h"
#include "NVGPUKernels/GemmProfiler.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define PASS_NAME "homnvgpu-autotune"
#define DEBUG_TYPE PASS_NAME

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu {

#define GEN_PASS_DEF_HOMNVGPUAUTOTUNEPASS
#include "HOMNVGPU/HOMNVGPUAutotune.pdll.h.inc"
#include "HOMNVGPU/Passes.h.inc"

namespace {

static void profileMatmulImpl(PatternRewriter &rewriter, Operation *gemm_) {
  using namespace homnvgpu_kernel;
  auto gemm = dyn_cast<homnvgpu::MatmulOp>(gemm_);

  if (!gemm.getResult().getType().getElementType().isF16()) {
    llvm_unreachable("Not supported.");
  }

  auto A = gemm.getOperand0().getType().getShape();
  auto B = gemm.getOperand1().getType().getShape();

  // To-do: Use a dedicated logger to log this.
  for (auto i : A) {
    std::cerr << i << " ";
  }

  std::cerr << std::endl;

  for (auto i : B) {
    std::cerr << i << " ";
  }

  std::cerr << std::endl;

  auto M = A[0] * A[1], N = B[2], K = A[2];
  auto alpha = gemm.getAlpha().convertToFloat();
  auto beta = gemm.getBeta().convertToFloat();
  auto act = gemm.getAct();

  if (act != 0 && act != 1) {
    return;
  }

  static GemmProfiler profiler(M, N, K, act, alpha, beta);

  auto [bestIdx, bestSplitKFactor] =
      profiler.profile(M, N, K, act, alpha, beta);

  gemm.setKernelName(bestIdx + 1);
  gemm.setSplitKFactor(bestSplitKFactor);
}

struct HOMNVGPUAutotunePass
    : impl::HOMNVGPUAutotunePassBase<HOMNVGPUAutotunePass> {
  void runOnOperation() final;

  LogicalResult initialize(MLIRContext *ctx) override;

private:
  FrozenRewritePatternSet patterns;
};

LogicalResult HOMNVGPUAutotunePass::initialize(MLIRContext *ctx) {
  RewritePatternSet patternList(ctx);

  populateGeneratedPDLLPatterns(patternList);
  patternList.getPDLPatterns().registerRewriteFunction("profileMatmul",
                                                       profileMatmulImpl);
  patterns = std::move(patternList);
  return success();
}

void HOMNVGPUAutotunePass::runOnOperation() {
  (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
}

} // namespace
} // namespace homnvgpu
} // namespace hands_on_mlir
} // namespace mlir
