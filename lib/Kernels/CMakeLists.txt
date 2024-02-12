file(GLOB _sources *.cu)

add_library(HOMNVGPUKernels SHARED ${_sources})
