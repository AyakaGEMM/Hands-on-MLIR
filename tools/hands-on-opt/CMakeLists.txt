get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_tool(hands-on-opt hands-on-opt.cpp)

target_link_libraries(
  hands-on-opt
  PRIVATE ${dialect_libs}
          ${conversion_libs}
          HOMFuncTransforms
          HOMHOMNVGPUNVGPUTransforms
          MLIROptLib
          MLIRArithDialect
          MatMulCPUOptimization
          MLIRHOM
          MLIRHOMNVGPU
          WeightsEngine
          TosaHOMTransforms)

mlir_check_all_link_libraries(hands-on-opt)
