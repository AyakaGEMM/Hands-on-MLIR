func.func @main() {
    %A = memref.alloc() : memref<4x8xf32>
    %i = arith.constant 0 : index
    %j = arith.constant 1 : index
    %M = arith.constant 4 : i64
    %N = arith.constant 8 : i64
    %c = arith.constant 4.0 : f32

    linalg.fill ins(%c : f32) outs(%A : memref<4x8xf32>)

    %B = memref.cast %A : memref<4x8xf32> to memref<*xf32>

    func.call @printMemrefF32(%B) : (memref<*xf32>) -> ()
    func.call @print2DMatrixF32(%B) : (memref<*xf32>) -> ()

    func.call @fill2DRandomMatrixF32(%B) : (memref<*xf32>) -> ()
    func.call @printMemrefF32(%B) : (memref<*xf32>) -> ()
    func.call @print2DMatrixF32(%B) : (memref<*xf32>) -> ()

    func.call @fill2DIncMatrixF32(%B) : (memref<*xf32>) -> ()
    func.call @printMemrefF32(%B) : (memref<*xf32>) -> ()
    func.call @print2DMatrixF32(%B) : (memref<*xf32>) -> ()

    return
}

func.func private @print2DMatrixF32(memref<*xf32>)
func.func private @fill2DRandomMatrixF32(memref<*xf32>)
func.func private @fill2DIncMatrixF32(memref<*xf32>)
func.func private @printMemrefF32(memref<*xf32>)
