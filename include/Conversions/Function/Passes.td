include "mlir/Pass/PassBase.td"

def HOMToFuncPass : Pass<"hom-to-func", "mlir::func::FuncOp"> {
  let summary = "Lowering HOM to Func";
  let dependentDialects = ["::mlir::func::FuncDialect"];
}

def HOMNVGPUToFuncPass : Pass<"homnvgpu-to-func", "mlir::func::FuncOp"> {
  let summary = "Lowering HOMNVGPU to Func";
  let dependentDialects =
      ["::mlir::func::FuncDialect", "::mlir::arith::ArithDialect"];
}

def OptimizeMemoryPass : Pass<"hom-opti-mem", "mlir::func::FuncOp"> {
  let summary = "Lowering HOMNVGPU to Func";
  let dependentDialects =
      ["::mlir::func::FuncDialect", "::mlir::arith::ArithDialect"];
}

def ExtractInitFuncPass : Pass<"extract-init-func", "mlir::ModuleOp"> {
  let summary = "Lowering Stablehlo to HOM";
  let dependentDialects =
      ["::mlir::arith::ArithDialect", "::mlir::func::FuncDialect"];
}

def UnifyLLVMFuncInterfacePass
    : Pass<"unify-llvm-func-interface", "mlir::ModuleOp"> {
  let summary = "Lowering Stablehlo to HOM";
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}
