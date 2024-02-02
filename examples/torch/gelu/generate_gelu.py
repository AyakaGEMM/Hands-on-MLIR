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
    module = torch_mlir.compile(
        model, hidden_states, output_type="tosa", use_tracing=True
    )
with open("gelu.mlir", "w") as fl:
    print(module, file=fl, end="")
