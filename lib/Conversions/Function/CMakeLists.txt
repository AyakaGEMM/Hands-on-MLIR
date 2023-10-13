add_mlir_library(HOMFuncTransforms
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