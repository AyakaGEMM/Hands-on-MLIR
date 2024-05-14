find_package(Python3 REQUIRED)

execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../python/
  COMMAND
    ${CMAKE_COMMAND} -E env PYTHONPATH=${CUTLASS_LIBRARY_PACKAGE_DIR}
    ${Python3_EXECUTABLE} ${HANDS_ON_MLIR_SOURCE_DIR}/python/gemm_generator.py
    --operations "*" # To-do: make it configurable
    --build-dir ${PROJECT_BINARY_DIR} --curr-build-dir
    ${CMAKE_CURRENT_BINARY_DIR} --generator-target library --architectures
    "${CMAKE_CUDA_ARCHITECTURES}"
    # To-do: make it configurable
    --kernels "*_aa_*" --ignore-kernels "${CUTLASS_LIBRARY_IGNORE_KERNELS}"
    --kernel-filter-file "${KERNEL_FILTER_FILE}" --selected-kernel-list
    "${CUTLASS_LIBRARY_GENERATED_KERNEL_LIST_FILE}" --cuda-version
    "${CMAKE_CUDA_COMPILER_VERSION}" --log-level DEBUG
    --disable-cutlass-package-imports
  RESULT_VARIABLE cutlass_kernel_INSTANCE_GENERATION_RESULT
  OUTPUT_VARIABLE cutlass_kernel_INSTANCE_GENERATION_OUTPUT
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/cutlass_kernel_instance_generation.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/cutlass_kernel_instance_generation.log)

if(NOT cutlass_kernel_INSTANCE_GENERATION_RESULT EQUAL 0)
  message(
    FATAL_ERROR
      "Error generating library instances. See ${CMAKE_CURRENT_BINARY_DIR}/cutlass_kernel_instance_generation.log"
  )
endif()

message(
  STATUS
    "Completed generation of library instances. See ${CMAKE_CURRENT_BINARY_DIR}/cutlass_kernel_instance_generation.log for more information."
)

set(CUTLASS_KERNEL_MANIFEST_CMAKE_FILE
    ${CMAKE_CURRENT_BINARY_DIR}/generated/manifest.cmake)
if(EXISTS "${CUTLASS_KERNEL_MANIFEST_CMAKE_FILE}")
  include(${CUTLASS_KERNEL_MANIFEST_CMAKE_FILE})
else()
  message(
    STATUS
      "auto-generated library manifest cmake file (${CUTLASS_KERNEL_MANIFEST_CMAKE_FILE}) not found."
  )
endif()

add_library(GemmManifestAndProfiler STATIC GemmManifest.cu GemmProfiler.cu
                                           GemmRunner.cu)

target_compile_options(GemmManifestAndProfiler PRIVATE -fPIC)
target_link_libraries(GemmManifestAndProfiler cutlass_library_objs)
