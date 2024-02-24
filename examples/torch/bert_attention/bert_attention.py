import torch
import torch_mlir
from transformers import BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertAttention

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "Hello I'm a [MASK] model."
encoded_input = tokenizer(text, return_tensors="pt")
encoded_input_list = [
    torch.rand((1, 16, 4)),
    torch.zeros((1, 1, 1, 16), dtype=torch.float32),
]


class BertAttentionWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig().from_pretrained("bert-base-uncased")
        config.num_hidden_layers = 2
        config.hidden_size = 4
        config.num_attention_heads = 2
        self.attn = BertAttention(config)
        self.attn.self.query.bias.data.zero_()
        self.attn.self.key.bias.data.zero_()
        self.attn.self.value.bias.data.zero_()

    def forward(self, hidden, mask):
        return self.attn(hidden, mask)[0]


model = BertAttentionWrapper()
model.eval()

print(model(*encoded_input_list))

with torch.no_grad():
    module = torch_mlir.compile(
        model, encoded_input_list, output_type="TOSA", use_tracing=True
    )
    # output = model(*encoded_input_list)
with open("bert.mlir", "w") as fl:
    print(module, file=fl, end="")
