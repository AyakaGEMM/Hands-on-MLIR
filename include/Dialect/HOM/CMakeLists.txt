add_mlir_dialect(HOMOps hom)
set(LLVM_TARGET_DEFINITIONS HOMOps.td)
add_mlir_doc(HOMOps HOMDialect Dialects/ -gen-dialect-doc)