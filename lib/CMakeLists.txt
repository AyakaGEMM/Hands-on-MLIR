add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_mlir_library(static_mlir_async_runtime
  STATIC
  ${LLVM_MLIR_SOURCE_DIR}/lib/ExecutionEngine/AsyncRuntime.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ${LLVM_PTHREAD_LIB}
)

target_compile_definitions(static_mlir_async_runtime
  PRIVATE
  MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS
  )

  set(LLVM_OPTIONAL_SOURCES ExecutionEngine/HandsOnRunnerUtils.cpp)

add_mlir_library(hands_on_mlir_runner_utils
    SHARED
    ExecutionEngine/HandsOnRunnerUtils.cpp
  
    )
target_compile_definitions(hands_on_mlir_runner_utils PRIVATE hands_on_mlir_runner_utils_EXPORTS)