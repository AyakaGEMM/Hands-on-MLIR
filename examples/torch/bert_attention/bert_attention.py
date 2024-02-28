import torch
import torch_mlir
from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertAttention

torch.manual_seed(42)

hs = 768

encoded_input_list = [
    torch.rand((1, 64, hs)),
    torch.ones((1, 64), dtype=torch.int32),
]


class BertAttentionWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig().from_pretrained("bert-base-uncased")
        self.attn = BertAttention(config)
        self.attn.self.query.bias.data.zero_()
        self.attn.self.key.bias.data.zero_()
        self.attn.self.value.bias.data.zero_()
        self.attn.output.dense.bias.data.zero_()
        self.model = BertForMaskedLM(config)

    def forward(self, hidden, mask):
        mask = mask[:, None, None, :]
        mask = mask.to(torch.float32)
        mask = (1.0 - mask) * torch.finfo(torch.float32).min
        return self.model.bert.encoder(hidden, mask).last_hidden_state


model = BertAttentionWrapper()
model.eval()

with open("0.txt", "w") as fl:
    for i in encoded_input_list[0].view(-1):
        print(float(i), file=fl)
thing = model(*encoded_input_list)

print(thing)

with open("1.txt", "w") as fl:
    for i in thing.view(-1):
        print(float(i), file=fl)

with torch.no_grad():
    module = torch_mlir.compile(
        model, encoded_input_list, output_type="TOSA", use_tracing=True
    )
    # output = model(*encoded_input_list)
with open("bert_attn.mlir", "w") as fl:
    print(module, file=fl, end="")
