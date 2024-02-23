import torch
import torch_mlir

torch.manual_seed(42)


class A(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)


a = A()

x = torch.ones(2, 3, 100)

print(a(x))

module = torch_mlir.compile(a, x, output_type="tosa")
with open("linear.mlir", "w") as fl:
    print(module, file=fl, end="")
