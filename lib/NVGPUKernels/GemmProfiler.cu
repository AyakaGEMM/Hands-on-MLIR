#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/GemmManifest.h"
#include "NVGPUKernels/GemmProfiler.h"
#include "NVGPUKernels/GemmRunner.h"
#include "NVGPUKernels/Utils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <functional>
#include <numeric>
#include <tuple>

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu_kernel {

GemmProfiler::GemmProfiler(int64_t M, int64_t N, int64_t K, int64_t activation,
                           float alpha, float beta)
    : M_(M), N_(N), K_(K), activation_(activation), alpha_(alpha), beta_(beta),
      timingCache() {
  checkCudaErrors(cudaEventCreate(&start_));
  checkCudaErrors(cudaEventCreate(&stop_));

  a.rank = b.rank = tb.rank = c.rank = 3;
  a.descriptor = malloc(sizeof(StridedMemRefType<half, 3>));
  b.descriptor = malloc(sizeof(StridedMemRefType<half, 3>));
  tb.descriptor = malloc(sizeof(StridedMemRefType<half, 3>));
  c.descriptor = malloc(sizeof(StridedMemRefType<half, 3>));

  auto desA = static_cast<StridedMemRefType<half, 3> *>(a.descriptor);
  auto desB = static_cast<StridedMemRefType<half, 3> *>(b.descriptor);
  auto desTB = static_cast<StridedMemRefType<half, 3> *>(tb.descriptor);
  auto desC = static_cast<StridedMemRefType<half, 3> *>(c.descriptor);

  auto clearSize = [](StridedMemRefType<half, 3> &A) {
    A.sizes[0] = A.sizes[1] = A.sizes[2] = 0;
    A.offset = 0;
    A.data = nullptr;
  };

  clearSize(*desA);
  clearSize(*desB);
  clearSize(*desTB);
  clearSize(*desC);

  updateShape(a, 1, M, K);
  updateShape(b, 1, K, N);
  updateShape(tb, 1, N, K);
  updateShape(c, 1, M, N);

  updateSplitKFactor(K);
}

GemmProfiler::~GemmProfiler() {
  checkCudaErrors(cudaEventDestroy(start_));
  checkCudaErrors(cudaEventDestroy(stop_));

  auto freeMemref = [](C_UnrankedMemRefType &A) {
    auto desA = static_cast<StridedMemRefType<half, 3> *>(A.descriptor);

    checkCudaErrors(cudaFree(desA->data));
    free(desA);
  };

  freeMemref(a);
  freeMemref(b);
  freeMemref(tb);
  freeMemref(c);
}

void GemmProfiler::updateSplitKFactor(int32_t K) {
  auto maxK = std::max(std::min(16, K / 128), 1);

  splitKFactor.resize(maxK);

  for (int i = 1; i < maxK + 1; i++) {
    splitKFactor[i - 1] = i;
  }
}

float GemmProfiler::profileHelper(std::function<void()> runFn,
                                  const char *kernelName,
                                  float previousBestTime) {

  checkCudaErrors(cudaStreamSynchronize(nullptr));

  // Warm up
  for (int iter = 0; iter < 10; iter++) {
    runFn();
  }

  float totalTime = 0, batchTime;
  int64_t totalIter = 0;

  while (totalTime < 500) {
    checkCudaErrors(cudaEventRecord(start_));
    // Warm up
    for (int iter = 0; iter < 100; iter++) {
      runFn();
    }
    checkCudaErrors(cudaEventRecord(stop_));
    checkCudaErrors(cudaEventSynchronize(stop_));
    checkCudaErrors(cudaEventElapsedTime(&batchTime, start_, stop_));
    totalTime += batchTime;
    totalIter += 100;

    if (totalTime / totalIter / 1.2 > previousBestTime) {
      // To-do: Use a dedicated logger to log this.
      // std::cerr << "Skipping kernel(name: " << kernelName
      //           << ") since it is too slow. " << std::endl;
      break;
    }
  }

  // To-do: Use a dedicated logger to log this.
  // std::cerr << "Get kernel name: " << kernelName << " with time "
  //           << totalTime / totalIter << std::endl;

  return totalTime / totalIter;
}

void GemmProfiler::updateShape(C_UnrankedMemRefType &A, int64_t m, int64_t n,
                               int64_t k) {
  auto desA = static_cast<StridedMemRefType<half, 3> *>(A.descriptor);

  if (m * n * k > desA->offset) {
    checkCudaErrors(cudaFree(desA->data));
    checkCudaErrors(cudaMalloc(&(desA->data), m * n * k * sizeof(half)));
    desA->offset = m * n * k;
  }

  desA->sizes[0] = m;
  desA->sizes[1] = n;
  desA->sizes[2] = k;

  desA->strides[0] = n * k;
  desA->strides[1] = k;
  desA->strides[2] = 1;

  desA->basePtr = desA->data;
}

std::tuple<int64_t, int32_t> GemmProfiler::profile() {

  auto it = timingCache.find({M_, N_, K_, activation_});

  if (it != timingCache.end()) {
    auto [idx, splitKFactor] = it->second;
    // To-do: Use a dedicated logger to log this.
    std::cerr << "Cache hit, Idx = " << idx
              << ", split k factor = " << splitKFactor << std::endl;
    return it->second;
  }

  auto runNvteFn = [this]() {
    GemmNVTERunner<half> runner;
    auto status =
        runner.run(this->a.rank, this->a.descriptor, false, this->tb.rank,
                   this->tb.descriptor, true, this->c.rank, this->c.descriptor,
                   this->c.rank, this->c.descriptor, activation_, this->alpha_,
                   this->beta_);

    assert(status == cutlass::Status::kSuccess);
  };

  int64_t bestIdx = -1;
  float bestTime = profileHelper(runNvteFn, "nvte");
  auto basePerf = bestTime;
  int32_t bestSplitKFactor = 1;
  auto align = 8;

  while (M_ % align != 0) {
    align >>= 1;
  }
  while (N_ % align != 0) {
    align >>= 1;
  }
  while (K_ % align != 0) {
    align >>= 1;
  }

  if (activation_ != 0 && activation_ != 1) {
    llvm_unreachable("Not supported.");
  }

  for (size_t idx = 0; idx < manifest.size(); idx++) {
    auto kernel = manifest[idx].get();
    if (!kernel->isF16()) {
      continue;
    }
    if (kernel->getActivation() != activation_) {
      continue;
    }
    bool valid = false;
    auto alignInner = align;
    while (alignInner > 0 && !valid) {
      valid |= kernel->getGemmDescription().A.alignment == alignInner;
      alignInner >>= 1;
    }
    if (!valid) {
      continue;
    }

    for (auto k : splitKFactor) {
      auto runFn = [&kernel, this, k]() {
        auto error = kernel->run(
            this->a.rank, this->a.descriptor, this->b.rank, this->b.descriptor,
            this->c.rank, this->c.descriptor, this->c.rank, this->c.descriptor,
            this->alpha_, this->beta_, k);
        assert(error == Status::kSuccess);
      };

      auto kernelTime =
          profileHelper(runFn, kernel->getGemmDescription().name, bestTime);

      if (kernelTime < bestTime) {
        bestTime = kernelTime;
        bestIdx = idx;
        bestSplitKFactor = k;
      }
    }
  }

  // To-do: Use a dedicated logger to log this.
  std::cerr << "Profile for " << M_ << " " << N_ << " " << K_
            << " act: " << activation_ << " done. Best kernel "
            << (bestIdx == -1 ? "nvte"
                              : manifest[bestIdx]->getGemmDescription().name)
            << ", best split k factor " << bestSplitKFactor << ", best time "
            << bestTime << ", speedup compared to nvte " << basePerf / bestTime
            << ". " << std::endl;

  timingCache[{M_, N_, K_, activation_}] = {bestIdx, bestSplitKFactor};

  return {bestIdx, bestSplitKFactor};
}

std::tuple<int64_t, int32_t> GemmProfiler::profile(int64_t M, int64_t N,
                                                   int64_t K,
                                                   int64_t activation,
                                                   float alpha, float beta) {
  updateShape(a, 1, M, K);
  updateShape(b, 1, K, N);
  updateShape(tb, 1, N, K);
  updateShape(c, 1, M, N);

  updateSplitKFactor(K);
  activation_ = activation;

  M_ = M;
  N_ = N;
  K_ = K;

  alpha_ = alpha;
  beta_ = beta;

  return profile();
}

} // namespace homnvgpu_kernel
} // namespace hands_on_mlir
} // namespace mlir
