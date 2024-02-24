import torch
import torch_mlir
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention

torch.manual_seed(42)

encoded_input_list = [
    torch.rand((1, 16, 4)),
    torch.zeros((1, 16), dtype=torch.int32),
]


class BertAttentionWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig().from_pretrained("bert-base-uncased")
        config.num_hidden_layers = 2
        config.hidden_size = 4
        config.num_attention_heads = 2
        self.attn = BertSelfAttention(config)
        self.attn.query.bias.data.zero_()
        self.attn.key.bias.data.zero_()
        self.attn.value.bias.data.zero_()

    def forward(self, hidden, mask):
        mask = mask[:, None, None, :]
        mask = mask.to(torch.float32)
        mask = (1.0 - mask) * torch.finfo(torch.float32).min
        return self.attn(hidden, mask)[0]


model = BertAttentionWrapper()
model.eval()

print(encoded_input_list[0])

print(model(*encoded_input_list))

with torch.no_grad():
    module = torch_mlir.compile(
        model, encoded_input_list, output_type="TOSA", use_tracing=True
    )
    # output = model(*encoded_input_list)
with open("bert_self_attn.mlir", "w") as fl:
    print(module, file=fl, end="")
