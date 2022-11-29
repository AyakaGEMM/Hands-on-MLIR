func.func @main() {
    %A = memref.alloc() : memref<2048x2048xf64>
    %B = memref.alloc() : memref<2048x2048xf64>
    %C = memref.alloc() : memref<2048x2048xf64>
    %cf1 = arith.constant 1.0e+00 : f64
    linalg.fill ins(%cf1 : f64) outs(%A: memref<2048x2048xf64>)
    linalg.fill ins(%cf1 : f64) outs(%B: memref<2048x2048xf64>)
    linalg.fill ins(%cf1 : f64) outs(%C: memref<2048x2048xf64>)
    linalg.matmul ins(%A,%B:memref<2048x2048xf64>,memref<2048x2048xf64>) outs(%C:memref<2048x2048xf64>)
    return
}