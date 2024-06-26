#!/usr/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/lib/wsl/lib:/home/shared_folder/cudnn-linux-x86_64-9.0.0.312_cuda12-archive/lib

../../../build/bin/hands-on-opt --tosa-to-hom-pipeline --hom-fusion --hom-fp32-to-fp16 --hom-to-homnvgpu --homnvgpu-fusion bert.mlir > pre_tune.mlir

# ../../../build/bin/hands-on-opt --homnvgpu-legalize-gemm --tosa-layerwise-constant-fold --hom-serialize-weight --homnvgpu-to-func --hom-func-to-llvm-pipeline  pre_tune.mlir | \
# ../../../thirdparty/llvm-project/build/bin/mlir-translate --mlir-to-llvmir |\
# ../../../thirdparty/llvm-project/build/bin/llc > bert_nvgpu.s

../../../build/bin/hands-on-opt --homnvgpu-autotune --homnvgpu-legalize-gemm --tosa-layerwise-constant-fold --hom-serialize-weight --homnvgpu-to-func --hom-func-to-llvm-pipeline  pre_tune.mlir | \
../../../thirdparty/llvm-project/build/bin/mlir-translate --mlir-to-llvmir |\
../../../thirdparty/llvm-project/build/bin/llc > bert_autotune_nvgpu.s

clang++-17 bert_nvgpu.s -O3 -fPIC -shared -L../../../build/lib/ -lhands_on_mlir_execution_engine -lhands_on_mlir_nvgpu_runner_utils -L../../../thirdparty/llvm-project/build/lib -lLLVM-17 -std=gnu++17 -g -o libbert_nvgpu.so
clang++-17 bert_autotune_nvgpu.s -O3 -fPIC -shared -L../../../build/lib/ -lhands_on_mlir_execution_engine -lhands_on_mlir_nvgpu_runner_utils -L../../../thirdparty/llvm-project/build/lib -lLLVM-17 -std=gnu++17 -g -o libbert_autotune_nvgpu.so

# CUDNN_FRONTEND_LOG_INFO=1 CUDNN_FRONTEND_LOG_FILE=stderr CUDNN_LOGERR_DBG=3 CUDNN_LOGDEST_DBG=stderr \
# LSAN_OPTIONS=suppressions=../../../lsan.supp UBSAN_OPTIONS=suppressions=../../../ubsan.supp ASAN_OPTIONS=protect_shadow_gap=0,detect_odr_violation=0 ./run
