add_subdirectory(Dialect)
add_subdirectory(Conversions)
add_subdirectory(WeightsEngine)

add_mlir_library(
  static_mlir_async_runtime
  STATIC
  ${LLVM_MLIR_SOURCE_DIR}/lib/ExecutionEngine/AsyncRuntime.cpp
  EXCLUDE_FROM_LIBMLIR
  LINK_LIBS
  PUBLIC
  ${LLVM_PTHREAD_LIB})

target_compile_definitions(static_mlir_async_runtime
                           PRIVATE MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS)

set(LLVM_OPTIONAL_SOURCES ExecutionEngine/HandsOnRunnerUtils.cpp
                          ExecutionEngine/ExecutionEngine.cpp)

add_mlir_library(hands_on_mlir_runner_utils SHARED
                 ExecutionEngine/HandsOnRunnerUtils.cpp)

add_mlir_library(
  hands_on_mlir_execution_engine
  SHARED
  ExecutionEngine/ExecutionEngine.cpp
  LINK_COMPONENTS
  Core
  Support
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  WeightsEngine)

target_compile_definitions(hands_on_mlir_runner_utils
                           PRIVATE hands_on_mlir_runner_utils_EXPORTS)

if(ENABLE_CUDA)
  add_mlir_library(
    hands_on_mlir_nvgpu_runner_utils SHARED
    ExecutionEngine/HandsOnNVGPURunnerUtils.cu LINK_LIBS WeightsEngine)
  target_compile_options(hands_on_mlir_nvgpu_runner_utils PRIVATE -fexceptions)
  # add_subdirectory(NVGPUKernels)
endif()
