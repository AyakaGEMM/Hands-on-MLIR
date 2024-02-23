#!/usr/bin/bash

../../../../build/bin/hands-on-opt --tosa-to-hom-pipeline --hom-fusion --hom-serialize-weight  --hom-to-homnvgpu --homnvgpu-to-func --extract-init-func -convert-func-to-llvm -finalize-memref-to-llvm -convert-arith-to-llvm -unify-llvm-func-interface  layernorm.mlir |\
../../../../thirdparty/llvm-project/build/bin/mlir-translate --mlir-to-llvmir |\
../../../../thirdparty/llvm-project/build/bin/llc > layernorm_nvgpu.s

clang++-18 layernorm_nvgpu.s -fPIC -shared -L../../../../build/lib/ -lhands_on_mlir_execution_engine -lhands_on_mlir_nvgpu_runner_utils -L../../../../thirdparty/llvm-project/build/lib -lLLVM-18 -std=gnu++17 -g -o liblayernorm_nvgpu.so

clang++-18 layernorm.cu -g -debug -fsanitize=address,undefined -I../../../../include/ -I../../../../thirdparty/llvm-project/mlir/include/ -I../../../../thirdparty/llvm-project/llvm/include/ -I../../../../thirdparty/cutlass/include/ -I../../../../thirdparty/TransformerEngine/transformer_engine/common/include -I../../../../thirdparty/llvm-project/build/include/ -L./ -L../../../../build/lib/ -L../../../../thirdparty/llvm-project/build/lib -lLLVM-18 -lhands_on_mlir_runner_utils -llayernorm_nvgpu -lhands_on_mlir_nvgpu_runner_utils -lhands_on_mlir_execution_engine -L$CUDA_HOME/lib64 \
     -lcudart_static -Wl,-rpath,../../../../build/lib -Wl,-rpath,../../../../thirdparty/llvm-project/build/lib -Wl,-rpath,./ --cuda-gpu-arch=sm_89 -std=gnu++17 -o run

# LSAN_OPTIONS=suppressions=../../../../lsan.supp UBSAN_OPTIONS=suppressions=../../../../ubsan.supp ASAN_OPTIONS=protect_shadow_gap=0 ./run
