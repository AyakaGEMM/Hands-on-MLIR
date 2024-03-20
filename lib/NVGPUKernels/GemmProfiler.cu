#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/GemmManifest.h"
#include "NVGPUKernels/GemmProfiler.h"
#include "NVGPUKernels/GemmRunner.h"
#include "NVGPUKernels/Utils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstdint>
#include <functional>
#include <numeric>
#include <tuple>

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu_kernel {

GemmProfiler::GemmProfiler(int64_t M, int64_t N, int64_t K, float alpha,
                           float beta)
    : M_(M), N_(N), K_(K), alpha_(alpha), beta_(beta), timingCache() {
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
  auto maxK = std::max(std::min(64, K / 128), 1);

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

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  float totalTime = 0, batchTime;
  int64_t totalIter = 0;

  while (totalTime < 500) {
    checkCudaErrors(cudaEventRecord(start));
    // Warm up
    for (int iter = 0; iter < 100; iter++) {
      runFn();
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&batchTime, start, stop));
    totalTime += batchTime;
    totalIter += 100;

    if (totalTime / totalIter / 1.2 > previousBestTime) {
      // To-do: Use A dedicated logger to log this.
      // std::cerr << "Skipping kernel(name: " << kernelName
      //           << ") since it is too slow. " << std::endl;
      break;
    }
  }

  // To-do: Use A dedicated logger to log this.
  // std::cerr << "Get kernel name: " << kernelName << " with time "
  //           << totalTime / totalIter << std::endl;

  return totalTime / totalIter;
}

void GemmProfiler::updateShape(C_UnrankedMemRefType &A, int64_t m, int64_t n,
                               int64_t k) {
  auto desA = static_cast<StridedMemRefType<half, 3> *>(A.descriptor);

  auto oldTotalSize = std::accumulate(desA->sizes, desA->sizes + 3, 1,
                                      std::multiplies<int64_t>());

  if (m * n * k > oldTotalSize) {
    checkCudaErrors(cudaFree(desA->data));
    checkCudaErrors(cudaMalloc(&(desA->data), m * n * k * sizeof(half)));
  }

  desA->sizes[0] = m;
  desA->sizes[1] = n;
  desA->sizes[2] = k;

  desA->strides[0] = n * k;
  desA->strides[1] = k;
  desA->strides[2] = 1;

  desA->offset = 0;

  desA->basePtr = desA->data;
}

std::tuple<int64_t, int32_t> GemmProfiler::profile() {

  auto it = timingCache.find({M_, N_, K_});

  if (it != timingCache.end()) {
    auto [idx, splitKFactor] = it->second;
    // To-do: Use A dedicated logger to log this.
    std::cerr << "Cache hit, Idx = " << idx
              << ", split k factor = " << splitKFactor << std::endl;
    return it->second;
  }

  auto runNvteFn = [this]() {
    GemmNVTERunner<half> runner;
    auto status = runner.run(this->a.rank, this->a.descriptor, false,
                             this->tb.rank, this->tb.descriptor, true,
                             this->c.rank, this->c.descriptor, this->c.rank,
                             this->c.descriptor, 0, this->alpha_, this->beta_);

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

  // Ugly filter here.
  char alignStr[] = "_align0";

  for (size_t idx = 0; idx < manifest.size(); idx++) {
    auto kernel = manifest[idx].get();
    if (!kernel->isF16()) {
      continue;
    }
    bool valid = false;
    auto alignInner = align;
    while (alignInner > 0 && !valid) {
      alignStr[6] += alignInner;
      valid |= kernel->contains(alignStr);
      alignStr[6] -= alignInner;
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

  // To-do: Use A dedicated logger to log this.
  std::cerr << "Profile for " << M_ << " " << N_ << " " << K_
            << " done. Best kernel " << bestIdx << ", best split k factor "
            << bestSplitKFactor << ", best time " << bestTime
            << ", speedup compared to nvte " << basePerf / bestTime << ". "
            << std::endl;

  timingCache[{M_, N_, K_}] = {bestIdx, bestSplitKFactor};

  return {bestIdx, bestSplitKFactor};
}

std::tuple<int64_t, int32_t> GemmProfiler::profile(int64_t M, int64_t N,
                                                   int64_t K, float alpha,
                                                   float beta) {
  updateShape(a, 1, M, K);
  updateShape(b, 1, K, N);
  updateShape(tb, 1, N, K);
  updateShape(c, 1, M, N);

  updateSplitKFactor(K);

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
