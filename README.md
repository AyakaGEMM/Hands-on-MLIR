# Hands-on-MLIR

A simple project to optimize `linalg.matmul` using mlir framwork. Currently developing in progress. Feel free to creaet an issue if you have any suggestion or problem.

# What can it do?

Currently, this project can lower the `linalg.matmul` to `affine` dialect with tiling. Also, this project provide a simple benchmark to meassure the optimization's gFlops. However, it is not fast right now.(at about 2 gFlops compared to ~100 gFlops of mkl performance)

# To-do

+ explicit affine data packing mechanism. (`affineDataCopyGenerate` simply cannot work when the tensor shape is unknown. Maybe I should implement it myself.)
+ vector ld/st & compute.
+ And more...

# Install

## Install MLIR

Install it in your preferable way. This project should be compatible with the main branch of mlir.

## Install this project

If you didn't enable address sanitizer installing the mlir, please remove the following lines in CMakeLists.txt. (I'm to lazy to make it configurable)

```
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
```

Then use the following command to compile.

```
$ cd Hands-on-MLIR
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=/your/path/to/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=/your/path/to/llvm-project/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON
```

or you can use this setup in VSCode.

```
 "cmake.configureArgs": [
     "-DMLIR_DIR=/your/path/to/llvm-project/build/lib/cmake/mlir",
     "-DLLVM_DIR=/your/path/to/llvm-project/build/lib/cmake/llvm",
     "-DLLVM_ENABLE_ASSERTIONS=ON"
 ],
```

# Reference

+ [MLIR](https://github.com/llvm/llvm-project/)：抄了很多（
+ [buddy-mlir](https://github.com/buddy-compiler/buddy-mlir)：同样抄了很多（
+ [polymage-labs/mlirx](https://github.com/polymage-labs/mlirx)：版本太老了，很多都没法抄（
+ [Polyhedral Model 三篇](https://mp.weixin.qq.com/s?__biz=MzI3MDQ2MjA3OA==&mid=2247485130&idx=1&sn=a5773bf17e6854d1238b035366641bcc&chksm=ead1fbdbdda672cdf9b2480a431cef85e4d377d07f8c586a932adabd50656cbdcd7d891156bf&mpshare=1&scene=1&srcid=&sharer_sharetime=1569677798809&sharer_shareid=b33ef36fa0caf5cb82e76916516aa7df#rd)：知道多面体优化的基本概念。
