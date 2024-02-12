set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TosaToHOMTransforms)
add_public_tablegen_target(TosaToHOMTransformsPassIncGen)

add_mlir_pdll_library(TosaToHOMPDLLPatternsIncGen TosaToHOM.pdll
                      TosaToHOM.pdll.h.inc)
