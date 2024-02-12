file(GLOB sources_ *.cpp)

add_mlir_library(
  HOMToHOMNVGPUNVGPUTransforms
  ${sources_}
  DEPENDS
  HOMToHOMNVGPUNVGPUTransformsPassIncGen
  HOMToHOMNVGPUPDLLPatternsIncGen
  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRTransforms)
