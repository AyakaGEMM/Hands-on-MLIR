func.func @main() {
  func.call @test() : () -> ()
  return
}

func.func private @test() 