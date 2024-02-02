import torch
import torch_mlir
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "Hello I'm a [MASK] model."
encoded_input = tokenizer(text, return_tensors="pt")
encoded_input_list = [
    encoded_input["input_ids"],
    encoded_input["attention_mask"],
    encoded_input["token_type_ids"],
]


class BertWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig().from_pretrained("bert-base-uncased")
        config.num_hidden_layers = 2
        config.hidden_size = 24
        self.model = BertForMaskedLM(config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids, attention_mask, token_type_ids).logits


model = BertWrapper()

model.eval()

with torch.no_grad():
    module = torch_mlir.compile(
        model, encoded_input_list, output_type="tosa", use_tracing=True
    )
    # output = model(*encoded_input_list)
with open("bert.mlir", "w") as fl:
    print(module, file=fl, end="")
