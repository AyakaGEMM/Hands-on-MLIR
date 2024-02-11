#include "Conversions/Function/Passes.h"
#include "Conversions/MatMulCPUOptimize/Passes.h"
#include "Conversions/Tosa/Passes.h"
#include "HOM/Passes.h"
#include "HOMNVGPU/Passes.h"

namespace mlir {
namespace hands_on_mlir {
inline void registerAllHOMPasses() {
  registerMatMulCPUOptimizePass();
  hom::registerExtractInitFuncPass();
  hom::registerHOMFusionPass();
  hom::registerHOMSerializeWeightPass();
  hom::registerHOMToFuncPass();
  hom::registerTosaToHOMPass();
  hom::registerUnifyLLVMFuncInterfacePass();
  hom::registerTosaToHOMPipelines();
  homnvgpu::registerHOMNVGPUFusionPass();
}
} // namespace hands_on_mlir
} // namespace mlir
