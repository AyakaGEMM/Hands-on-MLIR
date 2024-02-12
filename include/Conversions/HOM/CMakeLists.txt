set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HOMToHOMNVGPUNVGPUTransforms)
add_public_tablegen_target(HOMToHOMNVGPUNVGPUTransformsPassIncGen)

add_mlir_pdll_library(HOMToHOMNVGPUPDLLPatternsIncGen HOMToHOMNVGPU.pdll
                      HOMToHOMNVGPU.pdll.h.inc)
