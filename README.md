# Hands-on-MLIR

WIP. Heavily developing in progress currently, so no document available. E2E bert is runnable with rtol at about 1e-3 with fp16 on 3090. Please see `examples/torch/bert`. The code is quite messy right now. Doesn't have time to clean up.

# Features

+ Lower torch model from TOSA to LLVM dialect.
+ End-to-end huggingface bert model support (Limited support)
    + E2E is limited supported now. Limitations are as follow (TE stands for transformer engine, HOM stands for hands on mlir project):
        1. `seqlen % 64 == 0` (TE limitation)
        2. `head_dim % 64 ==0` (TE limitation)
        3. `head_num > 1` (HOM limitation, the reshape pattern has some issue with `head_num==1`)
        4. fp16 only (HOM limitation, didn't write fp32 pass)
        5. padding mode only (HOM limitation)
        6. nvgpu only (HOM limitation)
        7. sm80+ only (TE limitation)
        8. Native Linux only (TE limitation)
        9. Static shape (HOM limitation)
+ Some simple fusion pass
    + GEMM + GELU fusion
    + Packed qkv bert attention
    + etc...
+ Autotuning cutlass
    + Only support GEMM, GEMM + GELU op with Row,Row,Row layout
    + The tilings are from cutlass official repo with some customization for gelu
    + Provide about 20% performance boost
    + Only support fp16 with fp32 acc
    + sm < 90 since I didn't generate sm90 cutlass kernel
    + Serial split k only

# To-do

+ Clean up code
+ Improve precision
+ More fusion pattern

# Pre-requirement

+ For nvcc, host compiler gcc >= 11. Clang is not tested.
+ For cpp code, Must use clang to compile (as new as possible for gnu++20 support)
    1. For _Float16 support (could be removed in the future)
    2. Used some other weird stuff. Simply just cannot compiled by gcc.
    3. C++ std=gnu++20 for template lambda support.
+ CUDNN > 8.9.7. If cudnn is not installed by package manager, you will also need to set env `CUDNN_PATH` for cudnn-frontend to find the correct cudnn location.
+ Only tested on sm86 and sm89. sm version lower than 80 is not supported.
+ Only tested on Linux. WSL is not supported.

# Install

## Clone submodules

```
# Wrote down from my memory, may not be correct.
git submodule init
git submodule update --recursive
```

## Install thirdparty

...

## Install python env

Install enssential python packages. Also, this project requires python venv since `transformer engine` needs torch with cuda, however, `torch-mlir` needs preview version of torch with cpu-only.

```
# Install script not finished.
pip install -r requirements.txt
pre-commit install
```

## Install MLIR

Install it in your preferable way. This project should be compatible with the main branch of mlir. Also, there is one under `thirdparty/llvm-project`, which is the one I'm currently working on. You can use that.  **Strongly recommend using lld to get faster linking speed.**

## Install this project

Use the following command to compile. **Strongly recommend using lld to get faster linking speed.**

```
$ cd Hands-on-MLIR
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=/your/path/to/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=/your/path/to/llvm-project/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_USE_LINKER=lld \
    -DENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=your sm version
```

or you can use this setup in VSCode.

```
 "cmake.configureArgs": [
     "-DMLIR_DIR=/your/path/to/llvm-project/build/lib/cmake/mlir",
     "-DLLVM_DIR=/your/path/to/llvm-project/build/lib/cmake/llvm",
     "-DLLVM_ENABLE_ASSERTIONS=ON",
     "-DLLVM_USE_LINKER=lld",
     "-DENABLE_CUDA=ON",
     "-DCMAKE_CUDA_ARCHITECTURES=your sm version ",
     // "-DLLVM_USE_SANITIZER=Address;Undefined" add this option if you want to enable the sanitizer. Also, maybe you should add it to llvm as well.
 ],
```

# Reference

+ [MLIR](https://github.com/llvm/llvm-project/)：抄了很多（
+ [buddy-mlir](https://github.com/buddy-compiler/buddy-mlir)：同样抄了很多（
+ [polymage-labs/mlirx](https://github.com/polymage-labs/mlirx)：版本太老了，很多都没法抄（
+ [Polyhedral Model 三篇](https://mp.weixin.qq.com/s?__biz=MzI3MDQ2MjA3OA==&mid=2247485130&idx=1&sn=a5773bf17e6854d1238b035366641bcc&chksm=ead1fbdbdda672cdf9b2480a431cef85e4d377d07f8c586a932adabd50656cbdcd7d891156bf&mpshare=1&scene=1&srcid=&sharer_sharetime=1569677798809&sharer_shareid=b33ef36fa0caf5cb82e76916516aa7df#rd)：知道多面体优化的基本概念。
