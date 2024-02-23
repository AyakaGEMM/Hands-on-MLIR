aux_source_directory(. DIR_LIB_SRCS)

add_mlir_library(
  HOMFP32ToFP16Transforms
  ${DIR_LIB_SRCS}
  DEPENDS
  HOMFP32ToFP16TransformsPassIncGen
  HOMFP32ToFP16PDLLPatternsIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms)
