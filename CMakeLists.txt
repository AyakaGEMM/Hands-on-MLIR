# ===- CMakeLists.txt - HANDS_ON_MLIR-mlir cmake root -----------------*- cmake
# -*-===//
#
# Configure the HANDS_ON_MLIR-mlir build.
#
# ===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

# -------------------------------------------------------------------------------
# Project setup and globals
# -------------------------------------------------------------------------------

project(Hands-on-MLIR LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------------------------
# Options and settings
# -------------------------------------------------------------------------------

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS
       "Build the LLVM tools. If OFF, just generate build targets." ON)

# -------------------------------------------------------------------------------
# MLIR/LLVM Configuration
# -------------------------------------------------------------------------------
find_package(MLIR REQUIRED CONFIG)

set(LLVM_MLIR_BINARY_DIR ${MLIR_DIR}/../../../bin)
set(LLVM_MLIR_SOURCE_DIR ${MLIR_DIR}/../../../../mlir)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# -------------------------------------------------------------------------------
# HANDS_ON_MLIR configuration
# -------------------------------------------------------------------------------

# HANDS_ON_MLIR project.
set(HANDS_ON_MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(HANDS_ON_MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(HANDS_ON_MLIR_INCLUDE_DIR ${HANDS_ON_MLIR_SOURCE_DIR}/include/)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${HANDS_ON_MLIR_BINARY_DIR})

set(HANDS_ON_MLIR_EXAMPLES
    OFF
    CACHE BOOL "Build examples")

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add HANDS_ON_MLIR files to the include path
include_directories(${HANDS_ON_MLIR_MAIN_INCLUDE_DIR})
include_directories(${HANDS_ON_MLIR_INCLUDE_DIR})
include_directories(${HANDS_ON_MLIR_INCLUDE_DIR}/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/Dialect)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${HANDS_ON_MLIR_SOURCE_DIR}/lib)
include_directories(${HANDS_ON_MLIR_SOURCE_DIR}/thirdparty/half-2.2.0/include)

# -------------------------------------------------------------------------------
# CUDA configuration
# -------------------------------------------------------------------------------

if(ENABLE_CUDA)
  add_compile_definitions(ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED YES)
  find_package(CUDAToolkit REQUIRED)
  include_directories(${HANDS_ON_MLIR_SOURCE_DIR}/thirdparty/cutlass/include)
  include_directories(
    ${HANDS_ON_MLIR_SOURCE_DIR}/thirdparty/cutlass/tools/library/include)
  include_directories(
    ${HANDS_ON_MLIR_SOURCE_DIR}/thirdparty/cutlass/tools/util/include)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

# -------------------------------------------------------------------------------
# Hardware detection
# -------------------------------------------------------------------------------

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_simd.cmake)
check_simd()

# -------------------------------------------------------------------------------
# Directory setup
# -------------------------------------------------------------------------------

add_subdirectory(cmake)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
