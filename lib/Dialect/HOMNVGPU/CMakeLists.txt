file(GLOB _sources *.cpp)

add_mlir_library(
  MLIRHOMNVGPU
  ${_sources}
  ADDITIONAL_HEADER_DIRS
  ${HANDS_ON_MLIR_INCLUDE_DIR}/Dialect/HOMNVGPU
  DEPENDS
  MLIRHOMNVGPUOpsIncGen
  HOMNVGPUTransformsPassIncGen
  HOMNVGPUFusionPDLLPatternsIncGen
  HOMNVGPUAutotunePDLLPatternsIncGen
  HOMNVGPULegalizeGemmPDLLPatternsIncGen
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRParser
  MLIRFuncDialect
  MLIRQuantDialect
  MLIRRewrite
  MLIRTransforms)
