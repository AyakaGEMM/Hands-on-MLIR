#ifndef CG_CGDIALECT_H
#define CG_CGDIALECT_H

#include "mlir/IR/Dialect.h"

#include "CpuGemm/IR/CGOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "CpuGemm/IR/CGOps.h.inc"

#endif // CG_CGDIALECT_H