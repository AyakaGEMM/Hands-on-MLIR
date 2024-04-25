#!/usr/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/lib/wsl/lib:/home/shared_folder/cudnn-linux-x86_64-9.0.0.312_cuda12-archive/lib

input_file=$1
file_name=${input_file%.mlir}
file_name=${file_name#./}
asm_file=${input_file%.mlir}.s
so_file=lib$file_name.so

echo "Processing $input_file"
echo "so_file $so_file"

../../../../build/bin/hands-on-opt --tosa-to-hom-pipeline --hom-fusion --hom-fp32-to-fp16 --hom-to-homnvgpu --homnvgpu-fusion --homnvgpu-legalize-gemm --tosa-layerwise-constant-fold --hom-serialize-weight --homnvgpu-to-func --hom-func-to-llvm-pipeline $input_file |\
../../../../thirdparty/llvm-project/build/bin/mlir-translate --mlir-to-llvmir |\
../../../../thirdparty/llvm-project/build/bin/llc > $asm_file

clang++-17 $asm_file -O3 -g -fPIC -shared -L../../../../build/lib/ -lhands_on_mlir_execution_engine -lhands_on_mlir_nvgpu_runner_utils -L../../../../thirdparty/llvm-project/build/lib -lLLVM-17 -std=gnu++17 -g -o $so_file

clang++-17 run_fp16.cu -g -O3 -I../../../../thirdparty/cutlass/tools/library/include -I../../../../include/ -I../../../../thirdparty/llvm-project/mlir/include/ -I../../../../thirdparty/TransformerEngine/transformer_engine/common/include -L../../../../thirdparty/TransformerEngine/ -I../../../../thirdparty/llvm-project/llvm/include/ -I../../../../thirdparty/cutlass/include/ -I../../../../thirdparty/llvm-project/build/include/ -L./ -L../../../../build/lib/ -L../../../../thirdparty/llvm-project/build/lib -ltransformer_engine -lLLVM-17 -lhands_on_mlir_runner_utils -lhands_on_mlir_nvgpu_runner_utils -lhands_on_mlir_execution_engine -ldl -lpthread -lrt -L$CUDA_HOME/lib64 \
     -lcudart_static -Wl,-rpath,../../../../build/lib -Wl,-rpath,../../../../thirdparty/llvm-project/build/lib -Wl,-rpath,../../../../thirdparty/TransformerEngine/  -Wl,-rpath,./ --cuda-gpu-arch=sm_86 -std=gnu++17 -o run

pattern="hom_linear_([0-9]+)_([0-9]+)_([0-9]+)\.mlir"
if [[ $input_file =~ $pattern ]]; then
    M="${BASH_REMATCH[1]}"
    N="${BASH_REMATCH[2]}"
    K="${BASH_REMATCH[3]}"
    # 输出提取的值
    echo "run_fp16.sh: M: $M, N: $N, K: $K"
fi

./run $M $N $K
