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
TIME_STEP = 40              # rnn time step  使语音长度，1帧10ms
# OUT_H_SIZE = 15           # rnn hidden size
LR = 0.001                  # learning rate
INPUT_CH = 42               # feature 维度， 图片宽度， 也有可能直接用
OUTPUT_CH1 = 6               # 卷积核的个数，相当于深度，也就是2D中第二个参数
OUTPUT_CH2 = 6               # 卷积核的个数，相当于深度，也就是2D中第二个参数
K1_SIZE = 3                 # 卷积核的尺寸
K2_SIZE = 2                   #


class VadNet(nn.Module):
    def __init__(self):
        super(VadNet, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=INPUT_CH, out_channels=OUTPUT_CH1, kernel_size=K1_SIZE,
            stride=1, padding=0, padding_mode='zeros', dilation=2, groups=1, bias=True)
        self.conv2 = nn.Conv1d(
            in_channels=OUTPUT_CH1, out_channels=OUTPUT_CH2, kernel_size=K2_SIZE,
            stride=2, padding=0, padding_mode='zeros', dilation=1, groups=1, bias=True)
        self.plat1 = nn.Flatten()
        self.fc1 = nn.Linear(108, 22)
        # self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(22, 1)         # 最后数据语音概率 数据集中分了三类，0 0.5 和 1， 所以我这边输出为这三类的概率，比较方便使用交叉熵
        self.out1 = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.plat1(x)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5)
        x = self.fc2(x)                 # 最后输出的是语音概率
        x = self.out1(x)
        return x

