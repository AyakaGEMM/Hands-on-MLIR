../../../build/bin/hands-on-opt --tosa-to-hom-pipeline --hom-serialize-weight --hom-to-func --extract-init-func -convert-func-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm -unify-llvm-func-interface linear.mlir | \
           ../../../thirdparty/llvm-project/build/bin/mlir-translate --mlir-to-llvmir | \
           ../../../thirdparty/llvm-project/build/bin/llc > linear.s

clang++-18 linear.s -fPIC -shared -L../../../build/lib/ -lhands_on_mlir_execution_engine -lhands_on_mlir_runner_utils -L../../../thirdparty/llvm-project/build/lib -lLLVM-18 -std=gnu++17 -g -o liblinear.so

clang++-18 test.cpp -fsanitize=address,undefined -I../../../include/ -I../../../thirdparty/llvm-project/mlir/include/ -I../../../thirdparty/llvm-project/llvm/include/ -I../../../thirdparty/llvm-project/build/include/ -L./ -L../../../build/lib/ -L../../../thirdparty/llvm-project/build/lib -lLLVM-18 -lhands_on_mlir_runner_utils -llinear -lhands_on_mlir_execution_engine -Wl,-rpath,../../../build/lib -Wl,-rpath,../../../thirdparty/llvm-project/build/lib -Wl,-rpath,./ -std=gnu++17 -o run

./run
