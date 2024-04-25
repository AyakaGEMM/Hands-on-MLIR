import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


@torch.no_grad()
def speed_test(model, input_list):
    print([(i.shape, i.dtype) for i in input_list])

    for _ in range(10):
        model(*input_list)

    a = torch.cuda.Event(True)
    b = torch.cuda.Event(True)
    a.record()

    for _ in range(1000):
        model(*input_list)

    b.record()

    torch.cuda.synchronize()

    time = a.elapsed_time(b)

    print(time / 1000)
