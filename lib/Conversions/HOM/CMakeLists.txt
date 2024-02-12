file(GLOB sources_ *.cpp)

add_mlir_library(
  HOMHOMNVGPUNVGPUTransforms
  ${sources_}
  DEPENDS
  HOMHOMNVGPUNVGPUTransformsPassIncGen
  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRTransforms)
