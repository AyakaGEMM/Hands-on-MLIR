add_mlir_dialect(HOMOps hom)
set(LLVM_TARGET_DEFINITIONS HOMOps.td)
add_mlir_doc(HOMOps HOMDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HOMTransforms)
add_public_tablegen_target(HOMTransformsPassIncGen)
