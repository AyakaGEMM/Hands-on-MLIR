import sys

sys.path.append(
    "/Users/pzzzzz/MyProjects/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir"
)

import torch
import torch_mlir

hidden_states = torch.rand((1, 10, 100))


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.LayerNorm(100)

    def forward(self, hidden_states):
        return torch.nn.functional.gelu(hidden_states)


model = Wrapper()

model.eval()

with torch.no_grad():
    a = torch_mlir.compile(model, hidden_states, output_type="tosa", use_tracing=True)
with open("gelu.mlir", "w") as fl:
    print(a, file=fl)
