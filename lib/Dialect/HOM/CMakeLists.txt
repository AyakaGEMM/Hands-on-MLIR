file(GLOB _sources
          *.cpp
    )

add_mlir_library(MLIRHOM
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${HANDS_ON_MLIR_INCLUDE_DIR}/Dialect/HOM

  DEPENDS
  MLIRHOMOpsIncGen
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
