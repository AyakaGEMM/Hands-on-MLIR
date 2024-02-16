file(GLOB sources_ *.cpp)

add_mlir_library(
  TosaToHOMTransforms
  ${sources_}
  DEPENDS
  TosaToHOMTransformsPassIncGen
  TosaToHOMPDLLPatternsIncGen
  WeightsEngine
  LINK_COMPONENTS
  Core
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  WeightsEngine)
