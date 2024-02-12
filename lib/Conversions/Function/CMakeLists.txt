file(GLOB sources_ *.cpp)

add_mlir_library(
  HOMFuncTransforms
  ${sources_}
  DEPENDS
  HOMFuncTransformsPassIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms)
