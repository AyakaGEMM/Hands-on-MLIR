import os
import sys

sys.path.append(os.path.realpath(".."))

import torch
import torch.cuda.nvtx as nvtx
from benchmark import speed_test
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def gen_input(bs, max_len):
    text = "Hello I'm a [MASK] model."
    encoded_input = tokenizer(text, return_tensors="pt")
    encoded_input_list = [
        encoded_input["input_ids"].expand(bs, -1),
        encoded_input["attention_mask"].expand(bs, -1),
        encoded_input["token_type_ids"].expand(bs, -1),
    ]

    encoded_input_list = [
        torch.concat(
            [
                i,
                (
                    torch.zeros(bs, max_len - i.shape[1], dtype=torch.int64)
                    if idx != 0
                    else (
                        torch.tensor(
                            [
                                [102 for k in range(max_len - i.shape[1])]
                                for j in range(bs)
                            ]
                        )
                        if idx == 2
                        else torch.ones(bs, max_len - i.shape[1], dtype=torch.int64)
                    )
                ),
            ],
            dim=-1,
        ).cuda()
        for idx, i in enumerate(encoded_input_list)
    ]

    return encoded_input_list


for model_mode in ["base", "large"]:
    for bs in [1, 8, 16, 32]:
        for max_len in [64, 128]:
            i = gen_input(bs, max_len)

            class BertWrapper(torch.nn.Module):
                def __init__(self):
                    with torch.cuda.amp.autocast():
                        super().__init__()
                        config = BertConfig().from_pretrained(
                            f"bert-{model_mode}-uncased"
                        )
                        self.model = BertForMaskedLM(config)

                def forward(self, input_ids, attention_mask, token_type_ids):
                    with torch.cuda.amp.autocast():
                        return self.model(
                            input_ids, attention_mask, token_type_ids
                        ).logits

            model = BertWrapper()
            model.eval()
            model = model.cuda()

            msg = f"model: {model_mode}, bs: {bs}, seq_len: {max_len}"

            print(msg)

            with nvtx.range(msg):
                speed_test(model, i)
