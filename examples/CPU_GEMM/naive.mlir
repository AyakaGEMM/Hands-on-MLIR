// Driver for sgemm matmul with initialization and GFLOPS reporting.
func.func @main() {
  %A = memref.alloc() : memref<2088x2048xf32>
  // Align %B and %C since these are shape cast to vector types.
  %B = memref.alloc() {alignment = 32} : memref<2048x2048xf32>
  %C = memref.alloc() {alignment = 32} : memref<2088x2048xf32>
  %C1 = memref.alloc() {alignment = 32} : memref<2088x2048xf32>

  %cf1 = arith.constant 1.00000e+01 : f32 // Large cf1 here to ensure beta is correct.

  %AA = memref.cast %A : memref<2088x2048xf32> to memref<*xf32>
  %BB = memref.cast %B : memref<2048x2048xf32> to memref<*xf32>
  %CC1 = memref.cast %C : memref<2088x2048xf32> to memref<*xf32>
  %CC2 = memref.cast %C1 : memref<2088x2048xf32> to memref<*xf32>

  %AAA = memref.cast %A : memref<2088x2048xf32> to memref<?x?xf32>
  %BBB = memref.cast %B : memref<2048x2048xf32> to memref<?x?xf32>
  %CCC1 = memref.cast %C : memref<2088x2048xf32> to memref<?x?xf32>
  %CCC2 = memref.cast %C1 : memref<2088x2048xf32> to memref<?x?xf32>

  func.call @fill2DRandomMatrixF32(%AA) : (memref<*xf32>) -> ()
  func.call @fill2DRandomMatrixF32(%BB) : (memref<*xf32>) -> ()

  linalg.fill ins(%cf1 : f32) outs(%C : memref<2088x2048xf32>)
  linalg.fill ins(%cf1 : f32) outs(%C1 : memref<2088x2048xf32>)
  func.call @matmul(%A, %B, %C) : (memref<2088x2048xf32>, memref<2048x2048xf32>, memref<2088x2048xf32>) -> ()
  func.call @mmatmul(%AAA, %BBB, %CCC1) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  func.call @validateF32WithRefMatmul(%AA, %BB, %CC2, %CC1) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xf32>) -> ()

  %reps = arith.constant 5 : index

  //warm up
  func.call @matmul(%A, %B, %C) : (memref<2088x2048xf32>, memref<2048x2048xf32>, memref<2088x2048xf32>) -> ()
  func.call @mmatmul(%AAA, %BBB, %CCC1) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  %t_start = func.call @rtclock() : () -> (f64)
  affine.for %ti = 0 to %reps {
    func.call @mmatmul(%AAA, %BBB, %CCC1) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    func.call @matmul(%A, %B, %C) : (memref<2088x2048xf32>, memref<2048x2048xf32>, memref<2088x2048xf32>) -> ()
  }
  %t_end = func.call @rtclock() : () -> (f64)
  %pC = memref.cast %C : memref<2088x2048xf32> to memref<*xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = memref.dim %C, %c0 : memref<2088x2048xf32>
  %N = memref.dim %C, %c1 : memref<2088x2048xf32>
  %K = memref.dim %A, %c1 : memref<2088x2048xf32>

  %t = arith.subf %t_end, %t_start : f64
  %f1 = arith.muli %M, %N : index
  %f2 = arith.muli %f1, %K : index
  // 2*M*N*K.
  %c2 = arith.constant 2 : index
  %f3 = arith.muli %c2, %f2 : index
  %num_flops = arith.muli %reps, %f3 : index
  %num_flops_i = arith.index_cast %num_flops : index to i64
  %num_flops_f = arith.sitofp %num_flops_i : i64 to f64
  %flops = arith.divf %num_flops_f, %t : f64
  func.call @printFlops(%flops) : (f64) -> ()

  return
}

#K_UB = affine_map<(d0) -> (480, d0 * -480 + 2048)>
#I_LB = affine_map<(d0) -> (d0 * 110)>
#I_UB = affine_map<(d0) -> (696, d0 * 110 + 110)>

func.func @matmul(%arg0: memref<2088x2048xf32>, %arg1: memref<2048x2048xf32>, %arg2: memref<2088x2048xf32>) {
  //linalg.matmul ins(%arg0,%arg1:memref<2088x2048xf32>,memref<2048x2048xf32>) outs(%arg2:memref<2088x2048xf32>)
return
}

func.func @mmatmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  linalg.matmul ins(%arg0,%arg1:memref<?x?xf32>,memref<?x?xf32>) outs(%arg2:memref<?x?xf32>)
return
}

func.func private @printFlops(f64)
func.func private @rtclock() -> f64
func.func private @print2DMatrixF32(memref<*xf32>)
func.func private @fill2DRandomMatrixF32(memref<*xf32>)
func.func private @fill2DIncMatrixF32(memref<*xf32>)
func.func private @printMemrefF32(memref<*xf32>)
func.func private @validateF32WithRefMatmul(memref<*xf32>,memref<*xf32>,memref<*xf32>,memref<*xf32>)