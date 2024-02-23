set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HOMFP32ToFP16Transforms)
add_public_tablegen_target(HOMFP32ToFP16TransformsPassIncGen)

add_mlir_pdll_library(HOMFP32ToFP16PDLLPatternsIncGen FP32toFP16.pdll
                      HOMFP32ToFP16.pdll.h.inc)
