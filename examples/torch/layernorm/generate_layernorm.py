import torch
import torch_mlir

hidden_states = torch.rand((1, 10, 100))


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.LayerNorm(10)
        self.l = torch.nn.Linear(100, 10, bias=False)

    def forward(self, hidden_states):
        return self.ln(self.l(hidden_states))


model = Wrapper()

model.eval()

with torch.no_grad():
    module = torch_mlir.compile(
        model, hidden_states, output_type="tosa", use_tracing=True
    )
with open("layernorm.mlir", "w") as fl:
    print(module, file=fl, end="")
