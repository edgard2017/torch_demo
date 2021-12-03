
import torch
import torch.nn as nn


class RnnCNet(nn.Module):
    def __init__(self):
        super(RnnCNet, self).__init__()
        self.rnn1 = nn.GRU(1, 32, 2, batch_first=True)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x = self.linear(x)
        return x


rnnc = RnnCNet()
inputs = torch.rand(1, 10, 1)
outputs = rnnc(inputs)
print(outputs.shape)


# print(outputs.shape)

# m = nn.Linear(20, 30)
# inputs = torch.randn(1, 1, 1, 1, 1, 1, 1, 128, 20)
# output = m(inputs)
# print(output.size())


# rnn = nn.RNNCell(10, 20)
# input = torch.randn(6, 4, 10)
# hx = torch.randn(4, 20)
# output = []
# for i in range(6):
#     hx = rnn(input[i], hx)
#     output.append(hx)
#
# print(hx)
# print(input[0].shape)



# rnn = nn.RNN(10, 20, 2, batch_first=True)
# input = torch.randn(3, 5, 10)
# h0 = torch.randn(2, 3, 20)
# output, hn = rnn(input, h0)
#
# print(output.shape)
# print(hn.shape)


