# Hands-on-MLIR

~~他妈的，mlir 真的是人能用的？我文档呢？我不到啊。~~

我摆了，直接 public 了。有好哥哥能教教我怎么创建 `AffineForOp` 吗？他那个循环体到底咋搞啊。那个循环体函数每个参数是干啥的啊。。。

还有好哥哥能教教我 `AffineMap` 是啥吗？我目前对他的理解是一种变换，但他怎么和 `AffineForOp` 交互的。。。

借用某大国知名高层的一句名言：

> 投降也不失为一种选择。

~~问过 chatgpt 了，没学会。。。mlir 变得太快了，给的代码只能知道大概意思是啥，编译不过去（~~

# Install

正常 MLIR 编译流程。这里给一个 vscode cmake 配置项，点一下 build 就能 build 了。记得开 Release，Debug 编不过去。

```
 "cmake.configureArgs": [
     "-DMLIR_DIR=/your/path/to/llvm-project/build/lib/cmake/mlir",
     "-DLLVM_DIR=/your/path/to/llvm-project/build/lib/cmake/llvm"
 ],
```