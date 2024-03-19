#pragma once

#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu_kernel {

class GemmProfiler {

  C_UnrankedMemRefType a, b, c, tb;

  float alpha_, beta_;

  int64_t M_, N_, K_;

  static void updateShape(C_UnrankedMemRefType &A, int64_t m, int64_t n,
                          int64_t k);

  std::vector<int32_t> splitKFactor;

  float
  profileHelper(std::function<void()> runFn, const char *kernelName,
                float previousBestTime = std::numeric_limits<float>::max());

  std::map<std::tuple<int64_t, int64_t, int64_t>, std::tuple<int64_t, int32_t>>
      timingCache;

  void updateSplitKFactor(int32_t K);

public:
  GemmProfiler() = delete;

  GemmProfiler(int64_t M, int64_t N, int64_t K, float alpha, float beta);
  ~GemmProfiler();

  std::tuple<int64_t, int32_t> profile();

  std::tuple<int64_t, int32_t> profile(int64_t M, int64_t N, int64_t K,
                                       float alpha, float beta);
};

} // namespace homnvgpu_kernel

} // namespace hands_on_mlir
} // namespace mlir
