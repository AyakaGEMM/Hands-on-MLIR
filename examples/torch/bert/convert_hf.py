import torch
import torch.amp
import torch_mlir
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

text = "Hello I'm a [MASK] model."
encoded_input = tokenizer(text, return_tensors="pt")


for name in ["bert-base-uncased", "bert-large-uncased"]:
    for bs in [32]:
        for len in [128]:
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
                            torch.zeros(bs, len - i.shape[1], dtype=torch.int64)
                            if idx != 0
                            else torch.tensor(
                                [
                                    [102 for k in range(len - i.shape[1])]
                                    for j in range(bs)
                                ]
                            )
                        ),
                    ],
                    dim=-1,
                )
                for idx, i in enumerate(encoded_input_list)
            ]

            class BertWrapper(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    config = BertConfig().from_pretrained(name)
                    self.model = BertForMaskedLM(config)

                def forward(self, input_ids, attention_mask, token_type_ids):
                    return self.model(input_ids, attention_mask, token_type_ids).logits

            model = BertWrapper()
            model.eval()

            output = model(*encoded_input_list)

            with torch.no_grad():
                module = torch_mlir.compile(
                    model, encoded_input_list, output_type="TOSA", use_tracing=True
                )

            with open(f"{name}_{bs}_{len}.mlir", "w") as fl:
                print(module, file=fl, end="")

            for idx in range(3):
                with open(f"{name}_{bs}_{len}_{idx}.txt", "w") as fl:
                    for i in encoded_input_list[idx].reshape(-1):
                        print(int(i), file=fl)

            with open(f"{name}_{bs}_{len}_3.txt", "w") as fl:
                for i in output.reshape(-1):
                    print(float(i), file=fl)
