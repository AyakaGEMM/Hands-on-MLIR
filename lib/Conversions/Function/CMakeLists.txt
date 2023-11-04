add_mlir_library(HOMFuncTransforms
  ExtractInitFunc.cpp
  FunctionCallUtils.cpp
  HOMToFunc.cpp
  UnifyLLVMFuncInterface.cpp

  DEPENDS
  HOMFuncTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
)