#include "NVGPUKernels/GemmRunner.h"
#include <cstring>

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu_kernel {

GemmOperationRunnerBase::~GemmOperationRunnerBase() {}
bool GemmOperationRunnerBase::contains(const char *str) {
  return strstr(description_.name, str) != nullptr;
}

} // namespace homnvgpu_kernel
} // namespace hands_on_mlir
} // namespace mlir
