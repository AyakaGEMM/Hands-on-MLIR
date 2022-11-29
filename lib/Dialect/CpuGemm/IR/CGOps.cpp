#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Transforms/InliningUtils.h"

#include "CpuGemm/IR/CGOps.h"

using namespace mlir;
using namespace hands_on_mlir::cg;

#include "CpuGemm/IR/CGOpsDialect.cpp.inc"

void CGDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "CpuGemm/IR/CGOps.cpp.inc"
        >();
}
