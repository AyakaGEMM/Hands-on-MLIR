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

include "mlir/Pass/PassBase.td"

def HOMFusionPass : Pass<"hom-fusion", "mlir::func::FuncOp"> {
  let summary = "HOM Fusion Pass";
  let dependentDialects = [
    "::mlir::hands_on_mlir::hom::HOMDialect", "::mlir::tosa::TosaDialect",
    "::mlir::pdl::PDLDialect", "::mlir::pdl_interp::PDLInterpDialect"
  ];
}

def HOMSerializeWeightPass
    : Pass<"hom-serialize-weight", "mlir::func::FuncOp"> {
  let summary = "Serialize Weight Pass";
  let dependentDialects = [
    "::mlir::hands_on_mlir::hom::HOMDialect", "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect"
  ];
}
