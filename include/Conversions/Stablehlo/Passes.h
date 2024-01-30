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

#ifndef HOM_CONVERSIONS_STABLEHLO_TRANSFORMS_PASSES_H
#define HOM_CONVERSIONS_STABLEHLO_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hands_on_mlir {
namespace hom {

#define GEN_PASS_DECL_STABLEHLOTOHOMPASS
#define GEN_PASS_REGISTRATION
// #include "Conversions/Stablehlo/Passes.h.inc"

} // namespace hom
} // namespace hands_on_mlir
} // namespace mlir

#endif // HOM_CONVERSIONS_STABLEHLO_TRANSFORMS_PASSES_H
