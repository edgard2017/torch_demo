
# 用来测试和理解RNN网络的基本使用方法
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyper Parameters
TIME_STEP = 40              # rnn time step  使语音长度，1帧10ms
# OUT_H_SIZE = 15           # rnn hidden size
LR = 0.001                  # learning rate
INPUT_CH = 42               # feature 维度， 图片宽度， 也有可能直接用
OUTPUT_CH1 = 12               # 卷积核的个数，相当于深度，也就是2D中第二个参数
OUTPUT_CH2 = 6               # 卷积核的个数，相当于深度，也就是2D中第二个参数
K1_SIZE = 2                 # 卷积核的尺寸
K2_SIZE = 2
BATCH_SIZE = 32


class Vad2Net(nn.Module):
    def __init__(self):
        super(Vad2Net, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=INPUT_CH, out_channels=OUTPUT_CH1, kernel_size=K1_SIZE,
            stride=1, padding=0, padding_mode='zeros', dilation=2, groups=1, bias=True)
        self.gru1 = nn.GRU(12, 12, 1, batch_first=True)
        self.plat1 = nn.Flatten()
        self.fc1 = nn.Linear(216, 22)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(22, 1)             # 最后数据语音概率 数据集中分了三类，0 0.5 和 1， 所以我这边输出为这三类的概率，比较方便使用交叉熵
        self.out1 = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.permute()
        x = self.plat1(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)  # 最后输出的是语音概率
        x = self.out1(x)
        return x



# vadnet1 = Vad2Net()
# inputs = torch.rand(BATCH_SIZE, TIME_STEP, INPUT_CH)
# inputs_con = inputs.permute(0, 2, 1)
# outputs = vadnet1(inputs_con)
# print(outputs.shape)




