import os
import sys

sys.path.append(os.path.realpath("../.."))

import torch
from benchmark import speed_test

torch.manual_seed(42)

for hidden in [768, 1024]:
    for m in [64]:
        for n in [hidden, hidden * 4]:
            for k in [hidden, hidden * 4]:

                if hidden != 1024:
                    continue

                if n != hidden or k != hidden * 4:
                    continue

                class LinearWithoutBias(torch.nn.Module):
                    def __init__(self) -> None:
                        super().__init__()
                        self.fc = torch.nn.Linear(k, n, bias=False)

                    def forward(self, x):
                        return self.fc(x)

                class LinearWithResidual(torch.nn.Module):
                    def __init__(self) -> None:
                        super().__init__()
                        self.fc = torch.nn.Linear(100, 10, bias=False)

                    def forward(self, x):
                        return self.fc(x) + x

                a = LinearWithoutBias().cuda().half()

                x = torch.ones(m, k).half().cuda()

                print(m, n, k)

                speed_test(a, [x])
