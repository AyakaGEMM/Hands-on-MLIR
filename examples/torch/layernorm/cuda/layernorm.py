import torch
import torch_mlir

torch.manual_seed(42)

hidden_states = torch.rand((1, 2, 10))


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.LayerNorm(10)

    def forward(self, hidden_states):
        return self.ln(hidden_states)


model = Wrapper()

model.eval()

print(hidden_states)

print(model(hidden_states))

with torch.no_grad():
    module = torch_mlir.compile(model, hidden_states, output_type="tosa")
with open("layernorm.mlir", "w") as fl:
    print(module, file=fl, end="")
