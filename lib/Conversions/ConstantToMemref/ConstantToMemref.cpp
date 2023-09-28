#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>

using namespace mlir;
using namespace vector;

namespace {

struct ValueToRange { // Work around for Value to Range convertion.
  SmallVector<Value> v_vector;
  ArrayRef<Value> ref;
  ValueRange vr;
  ValueToRange(Value &v) : v_vector(1, v), ref(v_vector), vr(ref) {}
  ValueToRange(const Value &v) : v_vector(1, v), ref(v_vector), vr(ref) {}
};

struct ConstantToMemref : public ConversionPattern {
  ConstantToMemref(MLIRContext *ctx)
      : ConversionPattern(stablehlo::ConstantOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (llvm::dyn_cast<stablehlo::ConstantOp>(op) == nullptr) {
      return failure();
    }

    auto loc = op->getLoc();
    auto a = op->getAttrs();
    auto b = a[0];

    auto name = b.getName().str();
    auto value = b.getValue().dyn_cast<DenseIntOrFPElementsAttr>();

    auto shape = value.getShapedType();
    auto element = value.getElementType();

    auto memrefType = UnrankedMemRefType::get(
        shape.getElementType(),
        shape.getNumElements() * shape.getElementTypeBitWidth());

    auto memref = rewriter.create<func::CallOp>(
        loc, getOrCreateAlloc(rewriter, op->getParentOfType<ModuleOp>(), shape),
        memrefType, ArrayRef<Value>());

    std::string fileName;

    if (element.isF32()) {
      static int fp32Idx = 0;
      fileName = "/home/pzzzzz/MyProjects/Hands-on-MLIR/examples/"
                 "contants2memref/" +
                 std::to_string(fp32Idx++) + ".txt";
      serializeWeightToDisk<APFloat>(value, fileName);
    } else if (element.isIntOrIndex()) {
      static int int32Idx = 0;
      fileName = "/home/pzzzzz/MyProjects/Hands-on-MLIR/examples/"
                 "contants2memref/" +
                 std::to_string(int32Idx++) + ".txt";
    }

    memref->setAttr("file", rewriter.getStringAttr(fileName));
    while (!op->getUses().empty()) {
      op->getUses().begin()->set(memref->getResult(0));
    }
    rewriter.replaceOp(op, memref);

    return success();
  }

  static FlatSymbolRefAttr getOrCreateAlloc(PatternRewriter &rewriter,
                                            ModuleOp module, ShapedType shape) {
    auto *context = module.getContext();
    if (module.lookupSymbol<func::FuncOp>("cudaAllocAndFill"))
      return SymbolRefAttr::get(context, "cudaAllocAndFill");

    // auto memrefType = UnrankedMemRefType::get(
    //     shape.getElementType(),
    //     shape.getNumElements() * shape.getElementTypeBitWidth());

    // auto memrefType = MemRefType::get(shape.getShape(),
    // shape.getElementType());

    // rewriter.getType<UnrankedMemRefType>(shape.getElementType(),
    //                                      shape.getNumElements() *
    //                                          shape.getElementTypeBitWidth());

    TypeRange inputType = {};
    TypeRange returnType = {rewriter.getF32Type()};

    auto funcFnType = rewriter.getFunctionType(
        {}, {UnrankedMemRefType::get(shape.getElementType(), 0)});

    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto cudaAllocFn = rewriter.create<func::FuncOp>(
        module->getLoc(), "cudaAllocAndFill", funcFnType);
    cudaAllocFn.setSymVisibility("private");
    return SymbolRefAttr::get(context, "cudaAllocAndFill");
  }

  template <class T>
  static void serializeWeightToDisk(DenseIntOrFPElementsAttr &value,
                                    const std::string &fileName) {
    auto shape = value.getShapedType();
    auto data = value.getValues<T>();
    auto dimSize = shape.getShape();
    // std::ofstream file(fileName);
    std::error_code EC;
    llvm::raw_fd_ostream out(fileName, EC);
    for (auto i : dimSize) {
      out << i << " ";
    }
    out << "\n";
    auto totalSize = value.getNumElements();
    for (int i = 0; i < totalSize; i++) {
      data[i].print(out);
    }
    out << "\n";
  }
}; // namespace
} // namespace

namespace {
struct ConstantToMemrefPass
    : public PassWrapper<ConstantToMemrefPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstantToMemrefPass)

  StringRef getArgument() const final { return "parse-stablehlo-constants"; }
  StringRef getDescription() const final {
    return "Parse `stablehlo.constant` value attribute.";
  }

  ConstantToMemrefPass() = default;
  ConstantToMemrefPass(const ConstantToMemrefPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, func::FuncDialect,
                    memref::MemRefDialect, math::MathDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ConstantToMemrefPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect, math::MathDialect>();
  target.addIllegalOp<linalg::MatmulOp>();

  RewritePatternSet patterns(context), simplify_patterns(context);
  patterns.add<ConstantToMemref>(context);
  FrozenRewritePatternSet frozenPatterns(std::move(simplify_patterns));

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir {
namespace hands_on_mlir {
void registerConstantToMemrefPass() {
  PassRegistration<ConstantToMemrefPass>();
}
} // namespace hands_on_mlir
} // namespace mlir