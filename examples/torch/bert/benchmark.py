import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

bs = 1

text = "Hello I'm a [MASK] model."
encoded_input = tokenizer(text, return_tensors="pt")
encoded_input_list = [
    encoded_input["input_ids"].expand(bs, -1),
    encoded_input["attention_mask"].expand(bs, -1),
    encoded_input["token_type_ids"].expand(bs, -1),
]

real_len = encoded_input_list[0].shape[1]

encoded_input_list = [
    torch.concat(
        [
            i,
            (
                torch.zeros(bs, 64 - i.shape[1], dtype=torch.int64)
                if idx != 0
                else torch.tensor(
                    [[102 for k in range(64 - i.shape[1])] for j in range(bs)]
                )
            ),
        ],
        dim=-1,
    ).cuda()
    for idx, i in enumerate(encoded_input_list)
]

print([(i.shape, i.dtype) for i in encoded_input_list])


class BertWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig().from_pretrained("bert-base-uncased")
        self.model = BertForMaskedLM(config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids, attention_mask, token_type_ids).logits


model = BertWrapper()
model.eval()
model = model.cuda()
model = model.half()

for i in range(10):
    output = model(*encoded_input_list)

a = torch.cuda.Event(True)
b = torch.cuda.Event(True)
a.record()

with torch.no_grad():
    for i in range(1000):
        output = model(*encoded_input_list)

b.record()

torch.cuda.synchronize()

time = a.elapsed_time(b)

print(time / 1000 / 1000)
