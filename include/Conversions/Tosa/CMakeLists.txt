set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TosaHOMTransforms)
add_public_tablegen_target(TosaHOMTransformsPassIncGen)

add_mlir_pdll_library(TosaHOMPDLLPatternsIncGen TosaToHOM.pdll
                      TosaToHOM.pdll.h.inc)
