import torch
import torch_mlir
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertAttention

torch.manual_seed(42)

encoded_input_list = [
    torch.rand((1, 64, 128)),
    torch.zeros((1, 64), dtype=torch.int32),
]


class BertAttentionWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig().from_pretrained("bert-base-uncased")
        config.num_hidden_layers = 2
        config.hidden_size = 128
        config.num_attention_heads = 2
        self.attn = BertAttention(config)
        self.attn.self.query.bias.data.zero_()
        self.attn.self.key.bias.data.zero_()
        self.attn.self.value.bias.data.zero_()
        self.attn.output.dense.bias.data.zero_()

    def forward(self, hidden, mask):
        mask = mask[:, None, None, :]
        mask = mask.to(torch.float32)
        mask = (1.0 - mask) * torch.finfo(torch.float32).min
        return self.attn(hidden, mask)[0]


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
