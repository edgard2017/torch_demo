
# 用来测试和理解RNN网络的基本使用方法
import torch
import torch.nn as nn
import torch.nn.functional as F


BATCH_SIZE = 32     #
TIME_SEQ = 10      # 时间序列
NDIM = 2           # fearture 维度
ELOOP = 3
# BLOCK = 10
OUT_SIZE = 1
FIRLEN = 5

# w = torch.tensor([0.71123, 0.90514, 1.13258, 1.2236, 3.22766])


class CauNet(nn.Module):
    def __init__(self):
        super(CauNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2)
        # self.conv2 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2)
        self.fc = nn.Flatten()
        self.lc1 = nn.Linear((TIME_SEQ-1)*2, OUT_SIZE)

    def forward(self, x):
        # x = F.relu(self.lc1(x))
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = self.fc(x)
        x = self.lc1(x)
        return x


if __name__ == '__main__':
    # 简单判断网络是否正常
    rnnet = CauNet()
    inputs = torch.rand(BATCH_SIZE, NDIM, TIME_SEQ)
    outputs = rnnet(inputs)
    print(outputs.shape)

    train_w = torch.load("../train_dat/train_w.data")

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(rnnet.parameters(), lr=0.001)

    # load data
    train_data = torch.load("../train_dat/train_t.data")
    label_data = torch.load("../train_dat/train_l.data")

    trains_d = train_data.reshape(-1, BATCH_SIZE, TIME_SEQ, NDIM)   # 第一维度自动判定
    labels_l = label_data.reshape(-1, BATCH_SIZE, TIME_SEQ, NDIM)
    BLOCK = trains_d.size()[0]

    for epoch in range(ELOOP):
        loss_epoch = 0
        NUM = BLOCK * BATCH_SIZE
        for i in range(BLOCK):
            trains_d_s = torch.squeeze(trains_d[i, :, :, :])
            labels_l_s = torch.squeeze(labels_l[i, :, :, :])
            labels_l_y = labels_l_s[:, TIME_SEQ-1, NDIM-OUT_SIZE:NDIM]

            # 需要对数据进行维度变换
            trains_d_s_w = trains_d_s.permute(0, 2, 1)

            out = rnnet(trains_d_s_w)  # # torch.Size([1, 1])
            loss = loss_fn(out, labels_l_y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_epoch += loss.item()
        print("epoch:{},  loss:{}".format(epoch, loss_epoch/NUM))

# 如何去验证是否正确呢？
    # 保存网络
    save_path = 'CauNet.pth'
    torch.save(rnnet.state_dict(), save_path)       # 将训练好的模型保存下来
    params = rnnet.state_dict()                     # 获得模型的原始状态以及参数。
















