import sys

sys.path.append(
    "/Users/pzzzzz/MyProjects/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir"
)

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

x = torch.ones(1, 3, 100)

module = torch_mlir.compile(a, x, output_type="tosa")
with open("b.mlir", "w") as fl:
    print(module, file=fl)

print(a(x))
