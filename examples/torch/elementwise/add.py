import torch
import torch_mlir

torch.manual_seed(42)

a = torch.rand((3, 3, 3))
b = torch.rand((3, 1, 3))


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b


model = Wrapper()

model.eval()

print(a, b)

print(model(a, b))

with torch.no_grad():
    module = torch_mlir.compile(model, (a, b), output_type="tosa")
with open("add.mlir", "w") as fl:
    print(module, file=fl, end="")
