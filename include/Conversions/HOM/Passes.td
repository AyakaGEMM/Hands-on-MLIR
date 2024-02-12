include "mlir/Pass/PassBase.td"

def HOMToHOMNVGPUPass : Pass<"hom-to-homnvgpu", "mlir::func::FuncOp"> {
  let summary = "Lowering HOM to HOMNVGPU";
  let dependentDialects = ["::mlir::hands_on_mlir::homnvgpu::HOMNVGPUDialect"];
}
