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
