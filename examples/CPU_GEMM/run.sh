hands-on-opt -convert-linalg-to-loops \
		-lower-affine -convert-scf-to-cf -convert-vector-to-llvm \
		-convert-memref-to-llvm -convert-arith-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts optimized.mlir | \
        mlir-cpu-runner -O3 -e main \
        -entry-point-result=void \
        -shared-libs=../../../llvm-project/build/lib/libmlir_c_runner_utils.dylib \
        -shared-libs=../../../llvm-project/build/lib/libmlir_runner_utils.dylib \
        -shared-libs=../../build/lib/libhands_on_mlir_runner_utils.dylib