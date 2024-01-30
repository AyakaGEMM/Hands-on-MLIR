../../build/bin/hands-on-opt --tosa-layerwise-constant-fold --tosa-to-hom --hom-to-func --extract-init-func -convert-func-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm -unify-llvm-func-interface linear.mlir | \
           ../../thirdparty/llvm-project/build/bin/mlir-translate --mlir-to-llvmir | \
           ../../thirdparty/llvm-project/build/bin/llc > linear.s

clang++ linear.s -fPIC -shared -L../../build/lib/ -lhands_on_mlir_execution_engine -lhands_on_mlir_runner_utils  -o liblinear.so

clang++ test.cpp -I../../include/ -I../../thirdparty/llvm-project/mlir/include/ -I../../thirdparty/llvm-project/llvm/include/ -I../../thirdparty/llvm-project/build/include/ -L./ -L../../build/lib/ -lhands_on_mlir_runner_utils -llinear -lhands_on_mlir_execution_engine -Wl,-rpath,../../build/lib -Wl,-rpath,./ -stdlib=libc++ -std=c++17 -o run

./run
