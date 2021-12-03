
import torch
import torch.nn as nn
from rnn_net import RnnNet

BATCH_SIZE = 32     #
TIME_SEQ = 100      # 时间序列
NDIM = 10           # fearture 维度
ELOOP = 250
# BLOCK = 10
OUT_SIZE = 1
FIRLEN = 5
PERDECT_NUM = 100

train_w = torch.load("train_dat/train_w.data")

rc2net = RnnNet()  # 重新创建网络句柄
rc2net.load_state_dict(torch.load('RnnNet.pth'))  # 导入已经训练好的网络参数

error_loss = 0
for jk in range(PERDECT_NUM):
    inputs = torch.rand(NDIM)  # 测试网络
    inputs_net = inputs.reshape(1, 1, NDIM)
    outputs = rc2net(inputs_net)
    y_real = torch.zeros(NDIM)
    for j in range(NDIM - FIRLEN + 1):
        y_real[j + FIRLEN - 1] = torch.sum(inputs[j:j + 5] * train_w)
    error_loss += (outputs - y_real[NDIM - 1]) ** 2
# print(err)
print(error_loss / PERDECT_NUM)




