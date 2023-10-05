aux_source_directory(. DIR_LIB_SRCS)
set(LLVM_LINK_COMPONENTS Core Support)

add_llvm_library(WeightsEngine ${DIR_LIB_SRCS})

target_compile_options(WeightsEngine PRIVATE -fexceptions)