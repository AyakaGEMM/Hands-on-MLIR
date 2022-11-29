file(GLOB _sources
          IR/*.cpp
          Interfaces/*.cpp
          Canonicalize/*.cpp
          Transforms/*.cpp
    )

add_mlir_library(MLIRCG
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${HANDS_ON_MLIR_INCLUDE_DIR}/Dialect/CpuGemm

  DEPENDS
  MLIRCGOpsIncGen
  # MLIRCGTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRParser
  MLIRFuncDialect
  MLIRQuantDialect
  MLIRRewrite
  MLIRTransforms
)
