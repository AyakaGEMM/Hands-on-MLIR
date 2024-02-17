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

#ifndef HOM_CONVERSIONS_TOSA_TRANSFORMS_PASSES_H
#define HOM_CONVERSIONS_TOSA_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DECL_TOSATOHOMPASS
#define GEN_PASS_DECL_TOSACONSTANTFOLDINGPASS
#define GEN_PASS_REGISTRATION
#include "Conversions/Tosa/Passes.h.inc"

inline void registerTosaToHOMPipelines() {
  PassPipelineRegistration<>(
      "tosa-to-hom-pipeline",
      "Convert TOSA operators to hom with some optimization",
      [](OpPassManager &pm) {
        tosa::TosaLayerwiseConstantFoldPassOptions tosaConstFoldOption;
        tosaConstFoldOption.aggressiveReduceConstant = true;
        pm.addPass(
            tosa::createTosaLayerwiseConstantFoldPass(tosaConstFoldOption));
        pm.addPass(createTosaConstantFoldingPass());
        pm.addPass(
            tosa::createTosaLayerwiseConstantFoldPass(tosaConstFoldOption));
        pm.addPass(createTosaToHOMPass());
      });
}

} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir

#endif // HOM_CONVERSIONS_TOSA_TRANSFORMS_PASSES_H
