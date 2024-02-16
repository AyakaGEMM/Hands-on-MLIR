# Hands-on-MLIR

WIP. Heavily developing in progress currently, so no document available. Should be usable by the end of February 2024.

# To-do

+ ~~Integrate cutlass~~
+ End-to-end huggingface bert model support
+ Autotuning cutlass (If I have enough time)

# Install

## Install MLIR

Install it in your preferable way. This project should be compatible with the main branch of mlir. Also, there is one under `thirdparty/llvm-project`, which is the one I'm currently working on. You can use that.  **Strongly recommend using clang and lld to get faster compile speed.**

## Install this project

Use the following command to compile. **Strongly recommend using clang and lld to get faster compile speed.**

```
$ cd Hands-on-MLIR
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=/your/path/to/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=/your/path/to/llvm-project/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_USE_LINKER=lld # Strongly recommended to save the memory and linking time, so that we can compile with more threads, and could be linked faster. (On my local machine, could save the linking time from ~55s to sub 10s)
```

or you can use this setup in VSCode.

```
 "cmake.configureArgs": [
     "-DMLIR_DIR=/your/path/to/llvm-project/build/lib/cmake/mlir",
     "-DLLVM_DIR=/your/path/to/llvm-project/build/lib/cmake/llvm",
     "-DLLVM_ENABLE_ASSERTIONS=ON",
     "-DLLVM_USE_LINKER=lld",
     // "-DLLVM_USE_SANITIZER=Address;Undefined" add this option if you want to enable the sanitizer. Also, maybe you should add it to llvm as well.
 ],
```

# Reference

+ [MLIR](https://github.com/llvm/llvm-project/)：抄了很多（
+ [buddy-mlir](https://github.com/buddy-compiler/buddy-mlir)：同样抄了很多（
+ [polymage-labs/mlirx](https://github.com/polymage-labs/mlirx)：版本太老了，很多都没法抄（
+ [Polyhedral Model 三篇](https://mp.weixin.qq.com/s?__biz=MzI3MDQ2MjA3OA==&mid=2247485130&idx=1&sn=a5773bf17e6854d1238b035366641bcc&chksm=ead1fbdbdda672cdf9b2480a431cef85e4d377d07f8c586a932adabd50656cbdcd7d891156bf&mpshare=1&scene=1&srcid=&sharer_sharetime=1569677798809&sharer_shareid=b33ef36fa0caf5cb82e76916516aa7df#rd)：知道多面体优化的基本概念。
