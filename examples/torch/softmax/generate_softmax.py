import sys

sys.path.append(
    "/Users/pzzzzz/MyProjects/torch-mlir/build/tools/torch-mlir/python_packages/torch_mlir"
)

import torch
import torch_mlir
from transformers import BertConfig, BertForMaskedLM

hidden_states = torch.rand((1, 10))


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig().from_pretrained("bert-base-uncased")
        config.num_hidden_layers = 1
        config.hidden_size = 24
        self.model = BertForMaskedLM(config)

    def forward(self, hidden_states):
        return torch.nn.functional.softmax(hidden_states, dim=-1)


model = Wrapper()

model.eval()

with torch.no_grad():
    module = torch_mlir.compile(
        model, hidden_states, output_type="tosa", use_tracing=True
    )
    # output = model(*encoded_input_list)
with open("softmax.mlir", "w") as fl:
    print(module, file=fl, end="")
