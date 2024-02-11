add_mlir_dialect(HOMNVGPUOps homnvgpu)
set(LLVM_TARGET_DEFINITIONS HOMNVGPUOps.td)
add_mlir_doc(HOMNVGPUOps HOMNVGPUDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name HOMNVGPUTransforms)
add_public_tablegen_target(HOMNVGPUTransformsPassIncGen)

add_mlir_pdll_library(HOMNVGPUFusionPDLLPatternsIncGen HOMNVGPUFusion.pdll
                      HOMNVGPUFusion.pdll.h.inc)
