#include "Conversions/FP32toFP16/Passes.h"
#include "Conversions/Function/Passes.h"
#include "Conversions/HOM/Passes.h"
#include "Conversions/MatMulCPUOptimize/Passes.h"
#include "Conversions/Tosa/Passes.h"
#include "HOM/Passes.h"
#include "HOMNVGPU/Passes.h"

namespace mlir {
namespace hands_on_mlir {
inline void registerAllPasses() {
  registerExtractInitFuncPass();
  registerHOMFP32ToFP16Pass();
  registerHOMNVGPUToFuncPass();
  registerHOMToFuncPass();
  registerHOMToHOMNVGPUPass();
  registerMatMulCPUOptimizePass();
  registerUnifyLLVMFuncInterfacePass();
  registerHOMFuncToLLVMPipelines();
  hom::registerHOMFusionPass();
  hom::registerHOMSerializeWeightPass();
  hom::registerTosaToHOMPass();
  hom::registerTosaConstantFoldingPass();
  hom::registerTosaToHOMPipelines();
  homnvgpu::registerHOMNVGPUFusionPass();
  homnvgpu::registerHOMNVGPUAutotunePass();
  homnvgpu::registerHOMNVGPULegalizeGemmPass();
}
} // namespace hands_on_mlir
} // namespace mlir
