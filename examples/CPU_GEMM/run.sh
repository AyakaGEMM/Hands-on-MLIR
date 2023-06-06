../../build/bin/hands-on-opt --matmul-cpu-optimize --convert-linalg-to-affine-loops \
		-lower-affine -convert-scf-to-cf -convert-vector-to-llvm \
		-finalize-memref-to-llvm -convert-arith-to-llvm --convert-math-to-llvm \
		-convert-func-to-llvm -reconcile-unrealized-casts naive.mlir | \
        mlir-cpu-runner -O3 -e main \
        -entry-point-result=void \
        -shared-libs=../../../llvm-project/build/lib/libmlir_c_runner_utils.dylib \
        -shared-libs=../../../llvm-project/build/lib/libmlir_runner_utils.dylib \
        -shared-libs=../../build/lib/libhands_on_mlir_runner_utils.dylib