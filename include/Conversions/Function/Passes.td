include "mlir/Pass/PassBase.td"

def HOMToFuncPass : Pass<"hom-to-func", "mlir::func::FuncOp"> {
  let summary = "Lowering Stablehlo to HOM";
  let dependentDialects = ["::mlir::func::FuncDialect"];
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
