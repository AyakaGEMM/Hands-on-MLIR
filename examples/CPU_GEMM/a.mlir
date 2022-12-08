module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(0x4234640000000000 : f64) : f64
    %1 = llvm.mlir.constant(1.000000e+01 : f32) : f32
    %2 = llvm.mlir.constant(2088 : index) : i64
    %3 = llvm.mlir.constant(2048 : index) : i64
    %4 = llvm.mlir.constant(1 : index) : i64
    %5 = llvm.mlir.constant(4276224 : index) : i64
    %6 = llvm.mlir.null : !llvm.ptr<f32>
    %7 = llvm.getelementptr %6[4276224] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
    %8 = llvm.ptrtoint %7 : !llvm.ptr<f32> to i64
    %9 = llvm.call @malloc(%8) : (i64) -> !llvm.ptr<i8>
    %10 = llvm.bitcast %9 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %11 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %12 = llvm.insertvalue %10, %11[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %10, %12[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.mlir.constant(0 : index) : i64
    %15 = llvm.insertvalue %14, %13[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %2, %15[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %3, %16[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %3, %17[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %4, %18[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.mlir.constant(2048 : index) : i64
    %21 = llvm.mlir.constant(2048 : index) : i64
    %22 = llvm.mlir.constant(1 : index) : i64
    %23 = llvm.mlir.constant(4194304 : index) : i64
    %24 = llvm.mlir.null : !llvm.ptr<f32>
    %25 = llvm.getelementptr %24[4194304] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
    %26 = llvm.ptrtoint %25 : !llvm.ptr<f32> to i64
    %27 = llvm.mlir.constant(32 : index) : i64
    %28 = llvm.add %26, %27  : i64
    %29 = llvm.call @malloc(%28) : (i64) -> !llvm.ptr<i8>
    %30 = llvm.bitcast %29 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %31 = llvm.ptrtoint %30 : !llvm.ptr<f32> to i64
    %32 = llvm.mlir.constant(1 : index) : i64
    %33 = llvm.sub %27, %32  : i64
    %34 = llvm.add %31, %33  : i64
    %35 = llvm.urem %34, %27  : i64
    %36 = llvm.sub %34, %35  : i64
    %37 = llvm.inttoptr %36 : i64 to !llvm.ptr<f32>
    %38 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %39 = llvm.insertvalue %30, %38[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %40 = llvm.insertvalue %37, %39[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.mlir.constant(0 : index) : i64
    %42 = llvm.insertvalue %41, %40[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.insertvalue %20, %42[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.insertvalue %21, %43[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.insertvalue %21, %44[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.insertvalue %22, %45[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.mlir.constant(2088 : index) : i64
    %48 = llvm.mlir.constant(2048 : index) : i64
    %49 = llvm.mlir.constant(1 : index) : i64
    %50 = llvm.mlir.constant(4276224 : index) : i64
    %51 = llvm.mlir.null : !llvm.ptr<f32>
    %52 = llvm.getelementptr %51[4276224] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
    %53 = llvm.ptrtoint %52 : !llvm.ptr<f32> to i64
    %54 = llvm.mlir.constant(32 : index) : i64
    %55 = llvm.add %53, %54  : i64
    %56 = llvm.call @malloc(%55) : (i64) -> !llvm.ptr<i8>
    %57 = llvm.bitcast %56 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %58 = llvm.ptrtoint %57 : !llvm.ptr<f32> to i64
    %59 = llvm.mlir.constant(1 : index) : i64
    %60 = llvm.sub %54, %59  : i64
    %61 = llvm.add %58, %60  : i64
    %62 = llvm.urem %61, %54  : i64
    %63 = llvm.sub %61, %62  : i64
    %64 = llvm.inttoptr %63 : i64 to !llvm.ptr<f32>
    %65 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %66 = llvm.insertvalue %57, %65[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %67 = llvm.insertvalue %64, %66[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %68 = llvm.mlir.constant(0 : index) : i64
    %69 = llvm.insertvalue %68, %67[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %70 = llvm.insertvalue %47, %69[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.insertvalue %48, %70[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %72 = llvm.insertvalue %48, %71[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %73 = llvm.insertvalue %49, %72[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %74 = llvm.mlir.constant(2088 : index) : i64
    %75 = llvm.mlir.constant(2048 : index) : i64
    %76 = llvm.mlir.constant(1 : index) : i64
    %77 = llvm.mlir.constant(4276224 : index) : i64
    %78 = llvm.mlir.null : !llvm.ptr<f32>
    %79 = llvm.getelementptr %78[4276224] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
    %80 = llvm.ptrtoint %79 : !llvm.ptr<f32> to i64
    %81 = llvm.mlir.constant(32 : index) : i64
    %82 = llvm.add %80, %81  : i64
    %83 = llvm.call @malloc(%82) : (i64) -> !llvm.ptr<i8>
    %84 = llvm.bitcast %83 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %85 = llvm.ptrtoint %84 : !llvm.ptr<f32> to i64
    %86 = llvm.mlir.constant(1 : index) : i64
    %87 = llvm.sub %81, %86  : i64
    %88 = llvm.add %85, %87  : i64
    %89 = llvm.urem %88, %81  : i64
    %90 = llvm.sub %88, %89  : i64
    %91 = llvm.inttoptr %90 : i64 to !llvm.ptr<f32>
    %92 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %93 = llvm.insertvalue %84, %92[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %94 = llvm.insertvalue %91, %93[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %95 = llvm.mlir.constant(0 : index) : i64
    %96 = llvm.insertvalue %95, %94[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %97 = llvm.insertvalue %74, %96[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %98 = llvm.insertvalue %75, %97[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %99 = llvm.insertvalue %75, %98[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %100 = llvm.insertvalue %76, %99[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.mlir.constant(1 : index) : i64
    %102 = llvm.alloca %101 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    llvm.store %19, %102 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    %103 = llvm.bitcast %102 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>> to !llvm.ptr<i8>
    %104 = llvm.mlir.constant(2 : index) : i64
    %105 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %106 = llvm.insertvalue %104, %105[0] : !llvm.struct<(i64, ptr<i8>)> 
    %107 = llvm.insertvalue %103, %106[1] : !llvm.struct<(i64, ptr<i8>)> 
    %108 = llvm.mlir.constant(1 : index) : i64
    %109 = llvm.alloca %108 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    llvm.store %46, %109 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    %110 = llvm.bitcast %109 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>> to !llvm.ptr<i8>
    %111 = llvm.mlir.constant(2 : index) : i64
    %112 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %113 = llvm.insertvalue %111, %112[0] : !llvm.struct<(i64, ptr<i8>)> 
    %114 = llvm.insertvalue %110, %113[1] : !llvm.struct<(i64, ptr<i8>)> 
    %115 = llvm.mlir.constant(1 : index) : i64
    %116 = llvm.alloca %115 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    llvm.store %73, %116 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    %117 = llvm.bitcast %116 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>> to !llvm.ptr<i8>
    %118 = llvm.mlir.constant(2 : index) : i64
    %119 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %120 = llvm.insertvalue %118, %119[0] : !llvm.struct<(i64, ptr<i8>)> 
    %121 = llvm.insertvalue %117, %120[1] : !llvm.struct<(i64, ptr<i8>)> 
    %122 = llvm.mlir.constant(1 : index) : i64
    %123 = llvm.alloca %122 x !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    llvm.store %100, %123 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>>
    %124 = llvm.bitcast %123 : !llvm.ptr<struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>> to !llvm.ptr<i8>
    %125 = llvm.mlir.constant(2 : index) : i64
    %126 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
    %127 = llvm.insertvalue %125, %126[0] : !llvm.struct<(i64, ptr<i8>)> 
    %128 = llvm.insertvalue %124, %127[1] : !llvm.struct<(i64, ptr<i8>)> 
    llvm.call @fill2DRandomMatrixF32(%104, %103) : (i64, !llvm.ptr<i8>) -> ()
    llvm.call @fill2DRandomMatrixF32(%111, %110) : (i64, !llvm.ptr<i8>) -> ()
    %129 = llvm.mlir.constant(0 : index) : i64
    %130 = llvm.mlir.constant(2088 : index) : i64
    %131 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%129 : i64)
  ^bb1(%132: i64):  // 2 preds: ^bb0, ^bb5
    %133 = llvm.icmp "slt" %132, %130 : i64
    llvm.cond_br %133, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %134 = llvm.mlir.constant(0 : index) : i64
    %135 = llvm.mlir.constant(2048 : index) : i64
    %136 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%134 : i64)
  ^bb3(%137: i64):  // 2 preds: ^bb2, ^bb4
    %138 = llvm.icmp "slt" %137, %135 : i64
    llvm.cond_br %138, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %139 = llvm.mlir.constant(2048 : index) : i64
    %140 = llvm.mul %132, %139  : i64
    %141 = llvm.add %140, %137  : i64
    %142 = llvm.getelementptr %64[%141] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %1, %142 : !llvm.ptr<f32>
    %143 = llvm.add %137, %136  : i64
    llvm.br ^bb3(%143 : i64)
  ^bb5:  // pred: ^bb3
    %144 = llvm.add %132, %131  : i64
    llvm.br ^bb1(%144 : i64)
  ^bb6:  // pred: ^bb1
    %145 = llvm.mlir.constant(0 : index) : i64
    %146 = llvm.mlir.constant(2088 : index) : i64
    %147 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb7(%145 : i64)
  ^bb7(%148: i64):  // 2 preds: ^bb6, ^bb11
    %149 = llvm.icmp "slt" %148, %146 : i64
    llvm.cond_br %149, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    %150 = llvm.mlir.constant(0 : index) : i64
    %151 = llvm.mlir.constant(2048 : index) : i64
    %152 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb9(%150 : i64)
  ^bb9(%153: i64):  // 2 preds: ^bb8, ^bb10
    %154 = llvm.icmp "slt" %153, %151 : i64
    llvm.cond_br %154, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %155 = llvm.mlir.constant(2048 : index) : i64
    %156 = llvm.mul %148, %155  : i64
    %157 = llvm.add %156, %153  : i64
    %158 = llvm.getelementptr %91[%157] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %1, %158 : !llvm.ptr<f32>
    %159 = llvm.add %153, %152  : i64
    llvm.br ^bb9(%159 : i64)
  ^bb11:  // pred: ^bb9
    %160 = llvm.add %148, %147  : i64
    llvm.br ^bb7(%160 : i64)
  ^bb12:  // pred: ^bb7
    llvm.call @matmul(%10, %10, %14, %2, %3, %3, %4, %30, %37, %41, %20, %21, %21, %22, %57, %64, %68, %47, %48, %48, %49) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64) -> ()
    llvm.call @mmatmul(%10, %10, %14, %2, %3, %3, %4, %30, %37, %41, %20, %21, %21, %22, %57, %64, %68, %47, %48, %48, %49) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64) -> ()
    llvm.call @validateF32WithRefMatmul(%104, %103, %111, %110, %125, %124, %118, %117) : (i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>) -> ()
    llvm.call @matmul(%10, %10, %14, %2, %3, %3, %4, %30, %37, %41, %20, %21, %21, %22, %57, %64, %68, %47, %48, %48, %49) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64) -> ()
    llvm.call @mmatmul(%10, %10, %14, %2, %3, %3, %4, %30, %37, %41, %20, %21, %21, %22, %57, %64, %68, %47, %48, %48, %49) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64) -> ()
    %161 = llvm.call @rtclock() : () -> f64
    %162 = llvm.mlir.constant(0 : index) : i64
    %163 = llvm.mlir.constant(5 : index) : i64
    %164 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb13(%162 : i64)
  ^bb13(%165: i64):  // 2 preds: ^bb12, ^bb14
    %166 = llvm.icmp "slt" %165, %163 : i64
    llvm.cond_br %166, ^bb14, ^bb15
  ^bb14:  // pred: ^bb13
    llvm.call @mmatmul(%10, %10, %14, %2, %3, %3, %4, %30, %37, %41, %20, %21, %21, %22, %57, %64, %68, %47, %48, %48, %49) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64) -> ()
    llvm.call @matmul(%10, %10, %14, %2, %3, %3, %4, %30, %37, %41, %20, %21, %21, %22, %57, %64, %68, %47, %48, %48, %49) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64, !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64, i64, i64) -> ()
    %167 = llvm.add %165, %164  : i64
    llvm.br ^bb13(%167 : i64)
  ^bb15:  // pred: ^bb13
    %168 = llvm.call @rtclock() : () -> f64
    %169 = llvm.fsub %168, %161  : f64
    %170 = llvm.fdiv %0, %169  : f64
    llvm.call @printFlops(%170) : (f64) -> ()
    llvm.return
  }
  llvm.func @matmul(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<f32>, %arg8: !llvm.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<f32>, %arg15: !llvm.ptr<f32>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) {
    llvm.return
  }
  llvm.func @mmatmul(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr<f32>, %arg8: !llvm.ptr<f32>, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr<f32>, %arg15: !llvm.ptr<f32>, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg4, %5[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg6, %6[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %arg7, %8[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.insertvalue %arg10, %11[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %arg12, %12[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %arg11, %13[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg13, %14[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.insertvalue %arg14, %16[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.insertvalue %arg15, %17[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.insertvalue %arg16, %18[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %arg17, %19[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %arg19, %20[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %arg18, %21[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %arg20, %22[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.mlir.constant(1 : index) : i64
    %26 = llvm.extractvalue %7[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.extractvalue %23[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.extractvalue %15[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.mlir.constant(0 : index) : i64
    %30 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%29 : i64)
  ^bb1(%31: i64):  // 2 preds: ^bb0, ^bb8
    %32 = llvm.icmp "slt" %31, %26 : i64
    llvm.cond_br %32, ^bb2, ^bb9
  ^bb2:  // pred: ^bb1
    %33 = llvm.mlir.constant(0 : index) : i64
    %34 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%33 : i64)
  ^bb3(%35: i64):  // 2 preds: ^bb2, ^bb7
    %36 = llvm.icmp "slt" %35, %27 : i64
    llvm.cond_br %36, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %37 = llvm.mlir.constant(0 : index) : i64
    %38 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb5(%37 : i64)
  ^bb5(%39: i64):  // 2 preds: ^bb4, ^bb6
    %40 = llvm.icmp "slt" %39, %28 : i64
    llvm.cond_br %40, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %41 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %43 = llvm.mul %31, %42  : i64
    %44 = llvm.add %43, %39  : i64
    %45 = llvm.getelementptr %41[%44] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %46 = llvm.load %45 : !llvm.ptr<f32>
    %47 = llvm.extractvalue %15[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.extractvalue %15[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %49 = llvm.mul %39, %48  : i64
    %50 = llvm.add %49, %35  : i64
    %51 = llvm.getelementptr %47[%50] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %52 = llvm.load %51 : !llvm.ptr<f32>
    %53 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.mul %31, %54  : i64
    %56 = llvm.add %55, %35  : i64
    %57 = llvm.getelementptr %53[%56] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %58 = llvm.load %57 : !llvm.ptr<f32>
    %59 = llvm.intr.fma(%46, %52, %58)  : (f32, f32, f32) -> f32
    %60 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.extractvalue %23[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.mul %31, %61  : i64
    %63 = llvm.add %62, %35  : i64
    %64 = llvm.getelementptr %60[%63] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %59, %64 : !llvm.ptr<f32>
    %65 = llvm.add %39, %38  : i64
    llvm.br ^bb5(%65 : i64)
  ^bb7:  // pred: ^bb5
    %66 = llvm.add %35, %34  : i64
    llvm.br ^bb3(%66 : i64)
  ^bb8:  // pred: ^bb3
    %67 = llvm.add %31, %30  : i64
    llvm.br ^bb1(%67 : i64)
  ^bb9:  // pred: ^bb1
    llvm.return
  }
  llvm.func @printFlops(f64) attributes {sym_visibility = "private"}
  llvm.func @rtclock() -> f64 attributes {sym_visibility = "private"}
  llvm.func @print2DMatrixF32(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @fill2DRandomMatrixF32(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @fill2DIncMatrixF32(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @printMemrefF32(i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
  llvm.func @validateF32WithRefMatmul(i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>, i64, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
}

