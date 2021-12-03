

# 尝试复原下Rnnoise的vad网络

import torch
import torch.nn as nn
import torch.nn.functional as F

class RnnNet(nn.Module):
    def __init__(self):
        super(RnnNet, self).__init__()
        self.fc1 = nn.Linear(42, 24)
        self.gru1 = nn.GRU(24, 24)
        self.fc2 = nn.Linear(24, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gru1(x)
        x = self.fc2(x)
        return x


# Net information
rnnet = RnnNet()
print(rnnet)

## test net
inputs = torch.rand(5, 2, 42)
print(inputs.shape)
outputs = rnnet(inputs)
print("output = ", outputs)
# print("h = ", h)
