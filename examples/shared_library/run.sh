 clang++ test.cpp -fPIC -shared -o libtest.so
 
 ../../build/bin/hands-on-opt -convert-func-to-llvm -reconcile-unrealized-casts use_so.mlir | \
    ../../thirdparty/llvm-project/build/bin/mlir-translate -mlir-to-llvmir | \
    llc -opaque-pointers > test.s
    clang++ test.s -L. -ltest -Wl,-rpath=./ -o use_so

rm test.s