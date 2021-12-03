
# 用来测试和理解RNN网络的基本使用方法
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 32
OUTSIZE = 1
TIME_SEQ = 1000
NDIM = 161

# 网络是16K的 FFT结果来训练
class Vad1Net(nn.Module):
    def __init__(self):
        super(Vad1Net, self).__init__()
        self.lc1 = nn.Linear(NDIM, 24)
        self.dp1 = nn.Dropout()
        self.act1 = nn.Tanh()
        self.rnn1 = nn.GRU(24, 24, 1, batch_first=True)
        self.lc2 = nn.Linear(24, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.lc1(x)
        x = self.dp1(x)
        x = self.act1(x)
        x, _ = self.rnn1(x)
        x = self.lc2(x)
        x = self.act2(x)
        return x


# BATCH_SIZE = 32
# SEQ = 100
# Feat = 42
# vadnet1 = Vad1Net()
# inputs = torch.rand(BATCH_SIZE, SEQ, Feat)
# outputs = vadnet1(inputs)
# print(outputs)




