# 使用1D卷积，手写字体的训练，验证网络的有效性
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

## 获取数据集
from torch import squeeze


# Hyper Parameters
TIME_STEP = 28              # rnn time step
# OUT_H_SIZE = 15           # rnn hidden size
LR = 0.001                  # learning rate
INPUT_CH = 28               # feature 维度， 图片宽度
OUTPUT_CH = 6               # 卷积核的个数，相当于深度，也就是2D中第二个参数
KERNEL_S = 3                # 卷积核的尺寸
BATCH_SIZE = 20              # 训练一次块，每一个字自己训练
TEST_BATCH_SIZE = 1000      # 测试batch

class CauNet(nn.Module):
    def __init__(self):
        super(CauNet, self).__init__()
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
        # self.ax = nn.Tanh()
        self.conv2 = nn.Conv1d(
            in_channels=OUTPUT_CH,
            out_channels=2,
            kernel_size=KERNEL_S,
            stride=1,
            padding=0,
            padding_mode='zeros',
            dilation=1,
            groups=1,
            bias=True
        )
        self.plat1 = nn.Flatten()
        self.fc1 = nn.Linear(48, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.plat1(x)
        x = self.fc1(x)
        return x


transformation = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(
    'data/', train=True,
    transform=transformation,
    download=True)
test_dataset = torchvision.datasets.MNIST(
    'data/', train=False,
    transform=transformation,
    download=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=TEST_BATCH_SIZE,
    shuffle=False, num_workers=0)

test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()
test_image = torch.reshape(test_image, [TEST_BATCH_SIZE, 28, 28])

## Net information
CauNet = CauNet()
print(CauNet)
Input = torch.randn(BATCH_SIZE, INPUT_CH, TIME_STEP)
output = CauNet(Input)
print(output.shape)
loss_function = nn.CrossEntropyLoss()
# loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(CauNet.parameters(), lr=0.0005)

for epoch in range(1000):  # loop over the dataset multiple times
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = torch.reshape(inputs, [BATCH_SIZE, 28, 28])
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = CauNet(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if step % 500 == 499:  # print every 500 mini-batches  [batch, channel, height, width]
            with torch.no_grad():  # 对于with不太理解
                outputs = CauNet(test_image)  # [batch, 10]
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = torch.eq(predict_y, test_label).sum().item() / test_label.size(0)

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

