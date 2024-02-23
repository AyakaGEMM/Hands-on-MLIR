include "mlir/Pass/PassBase.td"

def HOMFP32ToFP16Pass : Pass<"hom-fp32-to-fp16", "mlir::func::FuncOp"> {
  let summary = "Lowering HOM to Func";
  let dependentDialects = ["::mlir::func::FuncDialect"];
}
