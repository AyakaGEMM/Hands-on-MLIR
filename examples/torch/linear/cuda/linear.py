import torch
import torch_mlir

torch.manual_seed(42)

hidden_state = 768

with torch.no_grad():
    for m in [64]:
        for n in [hidden_state, hidden_state * 4]:
            for k in [hidden_state, hidden_state * 4]:

                class LinearWithoutBias(torch.nn.Module):
                    def __init__(self) -> None:
                        super().__init__()
                        self.fc = torch.nn.Linear(k, n, bias=False)

                    def forward(self, x):
                        return self.fc(x)

                class LinearWithResidual(torch.nn.Module):
                    def __init__(self) -> None:
                        super().__init__()
                        self.fc = torch.nn.Linear(100, 10, bias=False)

                    def forward(self, x):
                        return self.fc(x) + x

                a = LinearWithoutBias()

                x = torch.ones(1, 64, k)

                module = torch_mlir.compile(a, x, output_type="tosa")
                with open(f"hom_linear_{m}_{n}_{k}.mlir", "w") as fl:
                    print(module, file=fl, end="")

                a = a.half()
                x = x.half()

                module = torch_mlir.compile(a, x, output_type="tosa")
                with open(f"iree_linear_{m}_{n}_{k}.mlir", "w") as fl:
                    print(module, file=fl, end="")
