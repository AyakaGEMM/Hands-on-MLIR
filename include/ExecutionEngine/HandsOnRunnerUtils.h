#ifndef HANDS_ON_MLIR_EXECUTIONENGINE_RUNNERUTILS_H
#define HANDS_ON_MLIR_EXECUTIONENGINE_RUNNERUTILS_H

#ifdef _WIN32 // Copied from official mlir project
#ifndef HANDS_ON_MLIR_RUNNERUTILS_EXPORT
#ifdef mlir_runner_utils_EXPORTS
// We are building this library
#define HANDS_ON_MLIR_RUNNERUTILS_EXPORT __declspec(dllexport)
#else
// We are using this library
#define HANDS_ON_MLIR_RUNNERUTILS_EXPORT __declspec(dllimport)
#endif // mlir_runner_utils_EXPORTS
#endif // MLIR_RUNNERUTILS_EXPORT
#else
// Non-windows: use visibility attributes.
#define HANDS_ON_MLIR_RUNNERUTILS_EXPORT __attribute__((visibility("default")))
#endif // _WIN32

#include <stddef.h>

extern "C" HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
fillRandomMatrix(float *, size_t, size_t);
extern "C" HANDS_ON_MLIR_RUNNERUTILS_EXPORT void
validateWithRefImpl(float *, float *, float *, float *, size_t, size_t, size_t);

#endif