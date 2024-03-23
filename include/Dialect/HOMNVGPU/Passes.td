include "mlir/Pass/PassBase.td"

def HOMNVGPUFusionPass : Pass<"homnvgpu-fusion", "mlir::func::FuncOp"> {
  let summary = "HOM Fusion Pass for NVIDIA GPU";
  let dependentDialects = [
    "::mlir::hands_on_mlir::hom::HOMDialect", "::mlir::tosa::TosaDialect",
    "::mlir::pdl::PDLDialect", "::mlir::pdl_interp::PDLInterpDialect"
  ];
}

def HOMNVGPULegalizeGemmPass
    : Pass<"homnvgpu-legalize-gemm", "mlir::func::FuncOp"> {
  let summary = "HOM Fusion Pass for NVIDIA GPU";
  let dependentDialects = [
    "::mlir::tosa::TosaDialect", "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect"
  ];
}

def HOMNVGPUAutotunePass : Pass<"homnvgpu-autotune", "mlir::func::FuncOp"> {
  let summary = "HOM Fusion Pass for NVIDIA GPU";
  let dependentDialects = [
    "::mlir::tosa::TosaDialect", "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect"
  ];
}
