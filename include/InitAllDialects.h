#include "HOM/HOMOps.h"
#include "HOMNVGPU/HOMNVGPUOps.h"

namespace mlir {
namespace hands_on_mlir {
inline void registerAllDialects(DialectRegistry &registry) {
  registry.insert<hom::HOMDialect, homnvgpu::HOMNVGPUDialect>();
}
} // namespace hands_on_mlir
} // namespace mlir
