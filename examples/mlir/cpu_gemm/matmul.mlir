func.func @main() {
  %A = memref.alloc() : memref<2088x2048xf32>
  // Align %B and %C since these are shape cast to vector types.
  %B = memref.alloc() {alignment = 32} : memref<2048x2048xf32>
  %C = memref.alloc() {alignment = 32} : memref<2088x2048xf32>
  linalg.matmul ins(%A, %B : memref<2088x2048xf32>, memref<2048x2048xf32>) outs(%C : memref<2088x2048xf32>)
  return
}
