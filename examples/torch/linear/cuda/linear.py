import torch
import torch_mlir

torch.manual_seed(42)

hidden_state = 100000


class LinearWithoutBias(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(hidden_state, 10, bias=False)

    def forward(self, x):
        return self.fc(x)


class LinearWithResidual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(100, 10, bias=False)

    def forward(self, x):
        return self.fc(x) + x


a = LinearWithoutBias()

x = torch.ones(2, 3, hidden_state)

module = torch_mlir.compile(a, x, output_type="tosa")
with open("linear.mlir", "w") as fl:
    print(module, file=fl, end="")

print(a(x))
