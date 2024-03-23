include "mlir/Pass/PassBase.td"

def TosaToHOMPass : Pass<"tosa-to-hom", "mlir::func::FuncOp"> {
  let summary = "Lower Tosa to the HOM dialect";
  let dependentDialects = [
    "::mlir::hands_on_mlir::hom::HOMDialect", "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect"
  ];
}

def TosaConstantFoldingPass
    : Pass<"hom-tosa-constant-folding", "mlir::func::FuncOp"> {
  let summary = "Lower Tosa to the HOM dialect";
  let dependentDialects = [
    "::mlir::tosa::TosaDialect", "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect"
  ];
}
