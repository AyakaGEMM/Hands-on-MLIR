#include "Kernels/CAPI.h"

extern "C" {
void MatmulDispatcher(HOMDtype dtype, void *A, void *B, void *C, float alpha,
                      float beta, const char *act) {}
}

template <typename T>
static auto MatmulRunner(T *A, T *B, T *C, float alpha, float beta,
                         const char *act) {}
