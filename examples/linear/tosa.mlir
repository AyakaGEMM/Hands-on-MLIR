module attributes {torch.debug_module_name = "A"} {
  func.func @forward(%arg0: tensor<1x3x2xf32>) -> (tensor<1x3x2xf32>, tensor<1x3x2xf32>) {
    %0 = "tosa.const"() <{value = dense<[[-0.344258487, 0.41527155], [0.623344957, -0.518753409]]> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
    %1 = "tosa.const"() <{value = dense<[[0.540610373, 0.586904228], [-0.165655658, 0.649556279]]> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
    %2 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %3 = "tosa.const"() <{value = dense<[[[-0.154929623, 0.142687559]]]> : tensor<1x1x2xf32>}> : () -> tensor<1x1x2xf32>
    %4 = "tosa.const"() <{value = dense<[[[0.614614487, 0.132341608]]]> : tensor<1x1x2xf32>}> : () -> tensor<1x1x2xf32>
    %5 = "tosa.transpose"(%1, %2) : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
    %6 = "tosa.reshape"(%5) <{new_shape = array<i64: 1, 2, 2>}> : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %7 = "tosa.matmul"(%arg0, %6) : (tensor<1x3x2xf32>, tensor<1x2x2xf32>) -> tensor<1x3x2xf32>
    %8 = "tosa.add"(%7, %3) : (tensor<1x3x2xf32>, tensor<1x1x2xf32>) -> tensor<1x3x2xf32>
    %9 = "tosa.transpose"(%0, %2) : (tensor<2x2xf32>, tensor<2xi32>) -> tensor<2x2xf32>
    %10 = "tosa.reshape"(%9) <{new_shape = array<i64: 1, 2, 2>}> : (tensor<2x2xf32>) -> tensor<1x2x2xf32>
    %11 = "tosa.matmul"(%arg0, %10) : (tensor<1x3x2xf32>, tensor<1x2x2xf32>) -> tensor<1x3x2xf32>
    %12 = "tosa.add"(%11, %4) : (tensor<1x3x2xf32>, tensor<1x1x2xf32>) -> tensor<1x3x2xf32>
    return %8, %12 : tensor<1x3x2xf32>, tensor<1x3x2xf32>
  }
}
