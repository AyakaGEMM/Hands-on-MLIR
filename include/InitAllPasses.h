#include "Conversions/Function/Passes.h"
#include "Conversions/MatMulCPUOptimize/Passes.h"
#include "Conversions/Tosa/Passes.h"
#include "HOM/Passes.h"
#include "HOMNVGPU/Passes.h"

namespace mlir {
namespace hands_on_mlir {
inline void registerAllHOMPasses() {
  registerMatMulCPUOptimizePass();
  registerExtractInitFuncPass();
  registerHOMToFuncPass();
  registerHOMNVGPUToFuncPass();
  registerUnifyLLVMFuncInterfacePass();
  hom::registerHOMFusionPass();
  hom::registerHOMSerializeWeightPass();
  hom::registerTosaToHOMPass();
  hom::registerTosaToHOMPipelines();
  homnvgpu::registerHOMNVGPUFusionPass();
}
} // namespace hands_on_mlir
} // namespace mlir
