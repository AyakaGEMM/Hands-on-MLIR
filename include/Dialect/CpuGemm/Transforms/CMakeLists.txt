set(LLVM_TARGET_DEFINITIONS CGTransforms.td)
mlir_tablegen(CGTransforms.h.inc -gen-rewriters)
add_public_tablegen_target(MLIRCGTransformsIncGen)