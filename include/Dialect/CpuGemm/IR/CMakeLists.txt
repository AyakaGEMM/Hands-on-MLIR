add_mlir_dialect(CGOps cg)
set(LLVM_TARGET_DEFINITIONS CGOps.td)
add_mlir_doc(CGOps CGDialect Dialects/ -gen-dialect-doc)