#pragma once

#include "NVGPUKernels/GemmRunner.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu_kernel {

class GemmManifest;

void initialize_all(GemmManifest &manifest);

class GemmManifest {
private:
  std::vector<std::unique_ptr<GemmOperationRunnerBase>> ops_;
  bool isInitialized;

public:
  void updateAllKernels() {
    if (!isInitialized) {
      init();
    }
  }

  GemmManifest() : isInitialized(false) {}

  auto &getKernel(int32_t idx) {
    if (!isInitialized) {
      init();
    }
    return ops_[idx];
  };

  auto &operator[](int idx) {
    if (!isInitialized) {
      init();
    }
    return ops_[idx];
  }

  void append(GemmOperationRunnerBase *operation) {
    ops_.emplace_back(operation);
  }

  void reserve(size_t size) { ops_.reserve(size); }

  auto size() {
    if (!isInitialized) {
      init();
    }
    return ops_.size();
  }

  void init() {
    if (!isInitialized) {
      initialize_all(*this);
      isInitialized = true;
    }
  }
};

extern GemmManifest manifest;

} // namespace homnvgpu_kernel
} // namespace hands_on_mlir
} // namespace mlir
