file(GLOB sources_ *.cpp)

add_mlir_library(
  HOMToFuncTransforms
  ${sources_}
  DEPENDS
  HOMToFuncTransformsPassIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms)
