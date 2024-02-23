#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"

#include "HOMNVGPU/HOMNVGPUOps.h"

using namespace mlir;
using namespace hands_on_mlir::homnvgpu;

#include "HOMNVGPU/HOMNVGPUOpsDialect.cpp.inc"

void HOMNVGPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "HOMNVGPU/HOMNVGPUOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "HOMNVGPU/HOMNVGPUOps.cpp.inc"

LogicalResult MatmulWithVarMeanOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto types = operands.getTypes();
  auto type0 = dyn_cast<RankedTensorType>(types[0]);
  auto type1 = dyn_cast<RankedTensorType>(types[1]);
  auto type2 = dyn_cast<RankedTensorType>(types[2]);

  if (type0 && type1 && type2 &&
      type0.getElementType() == type1.getElementType() &&
      type0.getElementType() == type2.getElementType()) {
    auto shape0 = type0.getShape();
    auto shape1 = type1.getShape();
    auto shape2 = type2.getShape();

    if (shape0.size() == shape1.size() && shape0.size() == shape2.size() &&
        shape0.size() == 3 && shape0[2] == shape1[1]) {
      auto resultType = RankedTensorType::get({std::max(shape0[0], shape2[0]),
                                               std::max(shape0[1], shape2[1]),
                                               std::max(shape1[2], shape2[2])},
                                              type0.getElementType());
      auto varType = RankedTensorType::get(
          {resultType.getShape()[0] * resultType.getShape()[1]},
          type0.getElementType());
      auto meanType = RankedTensorType::get(
          {resultType.getShape()[0] * resultType.getShape()[1]},
          type0.getElementType());

      inferredReturnTypes.emplace_back(resultType);
      inferredReturnTypes.emplace_back(varType);
      inferredReturnTypes.emplace_back(meanType);
      return success();
    }
  }

  return failure();
}
