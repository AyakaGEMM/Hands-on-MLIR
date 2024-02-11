file(GLOB _sources *.cu)

add_library(HOMCUDAKernels SHARED ${_sources})
