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
