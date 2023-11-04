../../build/bin/hands-on-opt --tosa-layerwise-constant-fold --tosa-to-hom --hom-to-func -convert-func-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm -unify-llvm-func-interface tosa.mlir | \
    ../../thirdparty/llvm-project/build/bin/mlir-translate --mlir-to-llvmir | \
    ../../thirdparty/llvm-project/build_clang/bin/llc > test.s

clang++ test.s -fPIC -shared -o liblinear.so

g++ -fsanitize=address,undefined test.cpp -I../../include/ -I../../thirdparty/llvm-project/mlir/include/ -L./ -L../../build/lib/ -lhands_on_mlir_runner_utils -llinear -Wl,-rpath=./ -Wl,-rpath=../../build/lib -o run

./run
