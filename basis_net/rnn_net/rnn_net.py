
# 用来测试和理解RNN网络的基本使用方法
import torch
import torch.nn as nn
import torch.nn.functional as F


BATCH_SIZE = 32     #
TIME_SEQ = 20      # 时间序列
NDIM = 10           # fearture 维度
ELOOP = 10
# BLOCK = 10
OUT_SIZE = 1
FIRLEN = 5

# w = torch.tensor([0.71123, 0.90514, 1.13258, 1.2236, 3.22766])


class RnnNet(nn.Module):
    def __init__(self):
        super(RnnNet, self).__init__()
        self.lc1 = nn.Linear(10, 5)
        self.rnn1 = nn.GRU(5, 10, 1, batch_first=True)
        self.lc2 = nn.Linear(10, OUT_SIZE)

    def forward(self, x):
        x = F.relu(self.lc1(x))
        x, _ = self.rnn1(x)
        x = self.lc2(x)
        return x


if __name__ == '__main__':
    # 简单判断网络是否正常
    rnnet = RnnNet()
    # inputs = torch.rand(1, 1, NDIM)
    # outputs = rnnet(inputs)
    # print(outputs.shape)

    train_w = torch.load("train_dat/train_w.data")

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(rnnet.parameters(), lr=0.001)

    # load data
    train_data = torch.load("train_dat/train_t.data")
    label_data = torch.load("train_dat/train_l.data")

    trains_d = train_data.reshape(-1, BATCH_SIZE, TIME_SEQ, NDIM)
    labels_l = label_data.reshape(-1, BATCH_SIZE, TIME_SEQ, NDIM)
    BLOCK = trains_d.size()[0]
    for epoch in range(ELOOP):
        loss_epoch = 0
        NUM = BLOCK*BATCH_SIZE
        for i in range(BLOCK):
            trains_d_s = torch.squeeze(trains_d[i, :, :, :])
            labels_l_s = torch.squeeze(labels_l[i, :, :, :])
            labels_l_y = labels_l_s[:, :, NDIM-OUT_SIZE:NDIM]
            optim.zero_grad()
            out = rnnet(trains_d_s)  # # torch.Size([1, 1])
            loss = loss_fn(out, labels_l_y)
            loss.backward()
            optim.step()
            loss_epoch += loss.item()
        print("epoch:{},  loss:{}".format(epoch, loss_epoch/NUM))

# 如何去验证是否正确呢？
    save_path = 'RnnNet.pth'
    torch.save(rnnet.state_dict(), save_path)       # 将训练好的模型保存下来
    params = rnnet.state_dict()                     # 获得模型的原始状态以及参数。

















