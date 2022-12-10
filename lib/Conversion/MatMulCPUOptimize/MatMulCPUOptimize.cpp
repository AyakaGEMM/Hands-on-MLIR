#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <iostream>

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

struct MatMulCPUOptimize : public ConversionPattern {
  MatMulCPUOptimize(MLIRContext *ctx)
      : ConversionPattern(linalg::MatmulOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Get input A, B, C.
    Value A = op->getOperand(0);
    Value B = op->getOperand(1);
    Value C = op->getOperand(2);

    // Create Constant
    const Value c0 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    const Value c1 =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    // Create M,N,K
    Value M = rewriter.create<memref::DimOp>(loc, A, c0);
    Value N = rewriter.create<memref::DimOp>(loc, C, c1);
    Value K = rewriter.create<memref::DimOp>(loc, B, c0);

    ValueToRange M_range(M), c0_range(c0);

    AffineForOp M_loop, N_loop, K_loop;

    M_loop = rewriter.create<AffineForOp>(
        loc, c0_range.vr, rewriter.getDimIdentityMap(), M_range.vr,
        rewriter.getDimIdentityMap(), 1, std::nullopt,
        [&](OpBuilder &builder, Location loc, Value im, ValueRange iterArgs) {
          ValueToRange N_range(N);
          N_loop = builder.create<AffineForOp>(
              loc, c0_range.vr, builder.getDimIdentityMap(), N_range.vr,
              builder.getDimIdentityMap(), 1, std::nullopt,
              [&](OpBuilder &builder, Location loc, Value in,
                  ValueRange iterArgs) {
                ValueToRange K_range(K);
                K_loop = builder.create<AffineForOp>(
                    loc, c0_range.vr, builder.getDimIdentityMap(), K_range.vr,
                    builder.getDimIdentityMap(), 1, std::nullopt,
                    [&](OpBuilder &builder, Location loc, Value ik,
                        ValueRange iterArgs) {
                      SmallVector<Value> load_A_mem_indices, load_B_mem_indices,
                          load_C_mem_indices;
                      load_A_mem_indices.push_back(im);
                      load_A_mem_indices.push_back(ik);
                      load_B_mem_indices.push_back(ik);
                      load_B_mem_indices.push_back(in);
                      load_C_mem_indices.push_back(im);
                      load_C_mem_indices.push_back(in);
                      Value a = builder.create<AffineLoadOp>(
                          loc, A, load_A_mem_indices);
                      Value b = builder.create<AffineLoadOp>(
                          loc, B, load_B_mem_indices);
                      Value c = builder.create<AffineLoadOp>(
                          loc, C, load_C_mem_indices);
                      Value resc = builder.create<math::FmaOp>(loc, a, b, c);
                      builder.create<AffineStoreOp>(loc, resc, C,
                                                    load_C_mem_indices);
                      builder.create<AffineYieldOp>(loc);
                    });
                Attribute K_Attr = rewriter.getStringAttr("K_loop");
                K_loop->setAttr("Dimension", K_Attr);
                builder.create<AffineYieldOp>(loc);
              });
          Attribute N_Attr = rewriter.getStringAttr("N_loop");
          N_loop->setAttr("Dimension", N_Attr);
          builder.create<AffineYieldOp>(loc);
        });

    Attribute M_Attr = rewriter.getStringAttr("M_loop");
    M_loop->setAttr("Dimension", M_Attr);

    interchangeLoops(N_loop, K_loop); // naive optimization
    interchangeLoops(M_loop, K_loop);
    interchangeLoops(M_loop, N_loop);

    // for (auto attr : M_loop->getAttrs()) {
    //   if (auto mapAttr = attr.getValue().dyn_cast<AffineMapAttr>()) {
    //     MutableAffineMap value = mapAttr.getValue();
    //     value.simplify();
    //   }
    // }

    rewriter.eraseOp(op);
    return success();
  }
}; // namespace
} // namespace

namespace {
struct MatMulCPUOptimizePass
    : public PassWrapper<MatMulCPUOptimizePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MatMulCPUOptimizePass)

  StringRef getArgument() const final { return "matmul-cpu-optimize"; }
  StringRef getDescription() const final {
    return "MatMul Optimization on CPU.";
  }

  MatMulCPUOptimizePass() = default;
  MatMulCPUOptimizePass(const MatMulCPUOptimizePass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, func::FuncDialect, memref::MemRefDialect,
                    math::MathDialect>();
  }
  void runOnOperation() final;

  // Copied from llvm-project
  template <typename AttributeT>
  void simplifyAndUpdateAttribute(Operation *op, StringAttr name,
                                  AttributeT attr) {
    auto &simplified = simplifiedAttributes[attr];
    if (simplified == attr)
      return;

    // This is a newly encountered attribute.
    if (!simplified) {
      // Try to simplify the value of the attribute.
      auto value = attr.getValue();
      auto simplifiedValue = simplify(value);
      if (simplifiedValue == value) {
        simplified = attr;
        return;
      }
      simplified = AttributeT::get(simplifiedValue);
    }

    // Simplification was successful, so update the attribute.
    op->setAttr(name, simplified);
  }

  IntegerSet simplify(IntegerSet set) { return simplifyIntegerSet(set); }

  /// Performs basic affine map simplifications.
  AffineMap simplify(AffineMap map) {
    MutableAffineMap mMap(map);
    mMap.simplify();
    return mMap.getAffineMap();
  }

  DenseMap<Attribute, Attribute> simplifiedAttributes;

  // Copy end.

  const size_t M_KERNEL_SIZE = 6;
  const size_t N_KERNEL_SIZE = 16;
  const int32_t K_BLOCK_SIZE = 1024;
  const int32_t M_BLOCK_SIZE = 384;
  const int32_t N_BLOCK_SIZE = 1024;
};
} // namespace

void getRootAffineForOp(func::FuncOp f,
                        std::vector<SmallVector<AffineForOp, 6>> *bands) {
  const char dim[] = "Dimension";
  StringRef root_loop_name("K_loop");
  for (AffineForOp forOp : f.getOps<AffineForOp>()) {
    auto stringAttr = forOp->getAttrOfType<StringAttr>(dim);
    if (!stringAttr)
      continue;
    auto loop_name = stringAttr.getValue();
    if (loop_name.equals(root_loop_name)) {
      SmallVector<AffineForOp, 6> band;
      getPerfectlyNestedLoops(band, forOp);
      bands->push_back(band);
    }
  }
}

AffineForOp getRootAffineForOpUnderIf(AffineForOp forOp) {
  AffineForOp res;
  AffineIfOp ifOp;
  int count = 2;
  forOp.walk([&](AffineIfOp op) { ifOp = op; }); // Only one if here.
  ifOp.walk([&](AffineForOp op) {
    if (count-- == 0) {
      res = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return res;
}

void MatMulCPUOptimizePass::runOnOperation() {
  MLIRContext *context = &getContext();
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect,
                         math::MathDialect>();
  target.addIllegalOp<linalg::MatmulOp>();

  RewritePatternSet patterns(context), simplify_patterns(context);
  patterns.add<MatMulCPUOptimize>(context);
  AffineApplyOp::getCanonicalizationPatterns(simplify_patterns, context);
  AffineForOp::getCanonicalizationPatterns(simplify_patterns, context);
  AffineIfOp::getCanonicalizationPatterns(simplify_patterns, context);
  memref::DimOp::getCanonicalizationPatterns(simplify_patterns, context);
  arith::ConstantIndexOp::getCanonicalizationPatterns(simplify_patterns,
                                                      context);
  FrozenRewritePatternSet frozenPatterns(std::move(simplify_patterns));

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();

  SmallVector<Operation *> opsToSimplify;
  getOperation().walk([&](Operation *op) {
    for (auto attr : op->getAttrs()) {
      if (auto mapAttr = attr.getValue().dyn_cast<AffineMapAttr>())
        simplifyAndUpdateAttribute(op, attr.getName(), mapAttr);
      else if (auto setAttr = attr.getValue().dyn_cast<IntegerSetAttr>())
        simplifyAndUpdateAttribute(op, attr.getName(), setAttr);
    }

    if (isa<AffineForOp, AffineIfOp, AffineApplyOp, memref::DimOp,
            arith::ConstantIndexOp>(op))
      opsToSimplify.push_back(op);
  });

  (void)applyOpPatternsAndFold(opsToSimplify, frozenPatterns, /*strict=*/true);

  std::vector<SmallVector<AffineForOp, 6>> bands;
  getRootAffineForOp(getOperation(), &bands);

  SmallVector<unsigned, 6> tile_sizes, // Here we use const parameters.
      kernel_tile_sizes;
  tile_sizes.push_back(K_BLOCK_SIZE);
  tile_sizes.push_back(N_BLOCK_SIZE);
  tile_sizes.push_back(M_BLOCK_SIZE);
  kernel_tile_sizes.push_back(M_KERNEL_SIZE);
  kernel_tile_sizes.push_back(N_KERNEL_SIZE);

  for (auto &band : bands) {
    Value A;
    SmallVector<AffineForOp, 6> tiled_nest;
    band[0].walk([&](memref::LoadOp op) { A = op.getMemRef(); });

    if (failed(tilePerfectlyNested(band, tile_sizes, &tiled_nest)))
      signalPassFailure();

    tiled_nest[0]->setAttr("Dimension", StringAttr::get(context, "K_BLOCK"));
    tiled_nest[1]->setAttr("Dimension", StringAttr::get(context, "N_BLOCK"));
    tiled_nest[2]->setAttr("Dimension", StringAttr::get(context, "M_BLOCK"));

    auto root_forOp = tiled_nest[0];
    band.clear();
    getPerfectlyNestedLoops(band, tiled_nest[3]);

    if (failed(separateFullTiles(band))) {
      std::cerr << "Separation Failed. " << std::endl;
      signalPassFailure();
    }

    auto new_start = getRootAffineForOpUnderIf(root_forOp);
    tiled_nest.clear();
    getPerfectlyNestedLoops(tiled_nest, new_start);
    interchangeLoops(new_start, tiled_nest[1]);
    interchangeLoops(new_start, tiled_nest[2]);
    interchangeLoops(tiled_nest[1], tiled_nest[2]);
    new_start->setAttr("Dimension", StringAttr::get(context, "K_KERNEL"));

    band.clear();
    getPerfectlyNestedLoops(band, tiled_nest[2]);
    band.pop_back();

    tiled_nest.clear();
    if (failed(tilePerfectlyNested(band, kernel_tile_sizes, &tiled_nest)))
      signalPassFailure();

    tiled_nest[0]->setAttr("Dimension", StringAttr::get(context, "M_CACHE"));
    tiled_nest[1]->setAttr("Dimension", StringAttr::get(context, "N_CACHE"));
    tiled_nest[2]->setAttr("Dimension", StringAttr::get(context, "M_KERNEL"));
    tiled_nest[3]->setAttr("Dimension", StringAttr::get(context, "N_KERNEL"));
    interchangeLoops(tiled_nest[3], new_start);
    interchangeLoops(tiled_nest[2], new_start);
    interchangeLoops(tiled_nest[2], tiled_nest[3]);

    auto m_loop_trip = getConstantTripCount(tiled_nest[2]),
         n_loop_trip = getConstantTripCount(tiled_nest[3]);
    if (m_loop_trip && n_loop_trip) {
      std::cerr << m_loop_trip.value() << " " << n_loop_trip.value()
                << std::endl;
    } else {
      std::cerr << "Not ok." << std::endl;
    }

    band.clear();
    getPerfectlyNestedLoops(band, new_start);
    (void)normalizeAffineFor(band[0]);
    (void)normalizeAffineFor(band[1]);
    (void)normalizeAffineFor(band[2]);
    // for (int i = band.size() - 1; i >= 0; i--) {
    //   auto forOp = band[i];
    //   if (failed(normalizeAffineFor(forOp))) {
    //     std::cerr << "Not ok." << std::endl;
    //   }
    // }

    // AffineCopyOptions copy_options = {false, 100, 100, 100, 512};
    // DenseSet<Operation *> fastBuf;

    // if (failed(affineDataCopyGenerate(tiled_nest[1].getBody()->begin(),
    //                                   std::prev(tiled_nest[1].getBody()->end()),
    //                                   copy_options, A, fastBuf))) {
    //   std::cerr << "Not ok." << std::endl;
    // }

    // if (failed(loopUnrollJamUpToFactor(tiled_nest[2], m_loop_trip.value())))
    // {
    //   std::cerr << "Not ok." << std::endl;
    // }
    //  if (failed(loopUnrollJamUpToFactor(tiled_nest[3], n_loop_trip.value())))
    //  {
    //    std::cerr << "Not ok." << std::endl;
    //  }
    //   if (failed(loopUnrollByFactor(new_start, 4))) {
    //     std::cerr << "Not ok." << std::endl;
    //   }

    // tiled_nest[2].setConstantLowerBound(0);
  }
}

namespace mlir {
namespace hands_on_mlir {
void registerMatMulCPUOptimizePass() {
  PassRegistration<MatMulCPUOptimizePass>();
}
} // namespace hands_on_mlir
} // namespace mlir