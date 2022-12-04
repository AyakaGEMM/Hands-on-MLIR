// Driver for sgemm matmul with initialization and GFLOPS reporting.
func.func @main() {
  %A = memref.alloc() : memref<2088x2048xf32>
  // Align %B and %C since these are shape cast to vector types.
  %B = memref.alloc() {alignment = 32} : memref<2048x2048xf32>
  %C = memref.alloc() {alignment = 32} : memref<2088x2048xf32>

  %cf1 = arith.constant 1.00000e+00 : f32

  linalg.fill ins(%cf1 : f32) outs(%A: memref<2088x2048xf32>)
  linalg.fill ins(%cf1 : f32) outs(%B: memref<2048x2048xf32>)

  %reps = arith.constant 5 : index

  %t_start = func.call @rtclock() : () -> (f64)
  affine.for %ti = 0 to %reps {
    linalg.fill ins(%cf1 : f32) outs(%C: memref<2088x2048xf32>)
    func.call @matmul_hop(%A, %B, %C) : (memref<2088x2048xf32>, memref<2048x2048xf32>, memref<2088x2048xf32>) -> ()
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

// This is a pre-tiled matmul loop nest matching the OpenBLAS/BLIS
// tiling strategy with L3 tiling being ignored:
// (i, j, k) -> (k, i, jj, ii, kk, jjR, iiR)
// With L3 tiling, this would have been:
// (i, j, k) -> (j, k, i, jj, ii, kk, jjR, iiR)
func.func @matmul_hop(%arg0: memref<2088x2048xf32>, %arg1: memref<2048x2048xf32>, %arg2: memref<2088x2048xf32>) {
  linalg.matmul ins(%arg0,%arg1:memref<2088x2048xf32>,memref<2048x2048xf32>) outs(%arg2:memref<2088x2048xf32>)
return
}

func.func private @printFlops(f64)
func.func private @rtclock() -> f64