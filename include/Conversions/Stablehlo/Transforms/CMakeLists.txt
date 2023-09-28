set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name StablehloHOMTransforms)
add_public_tablegen_target(StablehloMOHTransformsPassIncGen)

add_mlir_pdll_library(StablehloMOHPDLLPatternsIncGen
  StablehloToHOM.pdll
  StablehloToHOM.pdll.h.inc
)

add_mlir_library(StablehloHOMTransforms
  StablehloToHOM.cpp

  DEPENDS
  StablehloMOHTransformsPassIncGen
  StablehloMOHPDLLPatternsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
)
