## using conv1
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
# import torchsummary

import matplotlib.pyplot as plt

# Hyper Parameters
TIME_STEP = 10          # rnn time step
# OUT_H_SIZE = 15         # rnn hidden size
LR = 0.001              # learning rate
INPUT_CH = 1            # feature 维度
OUTPUT_CH = 2           # 卷积核的个数，相当于深度，也就是2D中第二个参数
KERNEL_S = 4            # 卷积核的尺寸
BATCH_SIZE = 1          # 训练一次块

# m = nn.Conv1d(16, 33, 3)
# input = torch.randn(20, 16, 50)
# output = m(input)

class CausalNet(nn.Module):
    def __init__(self):
        super(CausalNet, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=INPUT_CH,
            out_channels=OUTPUT_CH,
            kernel_size=KERNEL_S,
            stride=1,
            padding=0,
            padding_mode='zeros',
            dilation=1,
            groups=1,
            bias=True
        )

        self.linear1 = nn.Linear(TIME_STEP - KERNEL_S+1, INPUT_CH)

    def forward(self, x):
        x = self.conv1(x)
        x = self.linear1(x)
        return x

# causalnet = CausalNet()
# input = torch.randn(1, 3, TIME_STEP)
# output = causalnet(input)
# print(output.shape)


causalnet = CausalNet()
print(causalnet)

input = torch.randn(BATCH_SIZE, INPUT_CH, TIME_STEP)
output = causalnet(input)
print(output.shape)

optimizer = torch.optim.Adam(causalnet.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()
Loss_all = []
for step in range(1000):
    a = np.random.rand(1)
    start = (step + a) * np.pi
    end = start + np.pi
    # start, end = step , step * np.pi  # time range
    # use sin predicts cos
    steps = np.linspace(start, end,
                        BATCH_SIZE*INPUT_CH*TIME_STEP,
                        dtype=np.float32,
                        endpoint=False)  # float32 for converting torch FloatTensor
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
    # print(x.shape)
    x = x.reshape(BATCH_SIZE, INPUT_CH, TIME_STEP)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
    y = y.reshape(BATCH_SIZE, INPUT_CH, TIME_STEP)
    y_train = y[:, :, TIME_STEP-1]

    optimizer.zero_grad()
    prediction = causalnet(x)
    loss = loss_func(prediction, y_train)  # calculate loss
    if step % 5 == 4:
        Loss_all.append(loss.item())
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    print(loss)

plt.plot(Loss_all, lw=2)
plt.show()





















