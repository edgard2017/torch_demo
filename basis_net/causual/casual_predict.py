
import torch
import torch.nn as nn
from casual_net import CauNet

BATCH_SIZE = 32     #
TIME_SEQ = 10      # 时间序列
NDIM = 2           # fearture 维度
ELOOP = 300
# BLOCK = 10
OUT_SIZE = 1
FIRLEN = 5
PERDECT_NUM = 100
CAU_NUM = TIME_SEQ * NDIM

train_w = torch.load("train_dat/train_w.data")
rc2net = CauNet()  # 重新创建网络句柄
rc2net.load_state_dict(torch.load('CauNet.pth'))  # 导入已经训练好的网络参数

# inputs = torch.rand(NDIM * TIME_SEQ)  # 测试网络
# inputs_net = inputs.reshape(1, TIME_SEQ, NDIM)
# inputs_net = inputs_net.permute(0, 2, 1)
# outputs = rc2net(inputs_net)
# y_real = torch.zeros(NDIM * TIME_SEQ)
# for j in range(TIME_SEQ * NDIM - FIRLEN + 1):
#     y_real[j + FIRLEN - 1] = torch.sum(inputs[j:j + FIRLEN] * train_w)
# # print(err)
# print(inputs)
# print(outputs)
# print(y_real[TIME_SEQ * NDIM - 1])

error_loss = 0
for jk in range(PERDECT_NUM):
    inputs = torch.rand(CAU_NUM, requires_grad=False)  # 测试网络
    inputs_net = inputs.reshape(1, TIME_SEQ, NDIM)
    inputs_net = inputs_net.permute(0, 2, 1)
    outputs = rc2net(inputs_net)
    y_real = torch.zeros(CAU_NUM)
    for j in range(CAU_NUM - FIRLEN + 1):
        y_real[j + FIRLEN - 1] = torch.sum(inputs[j:j + FIRLEN] * train_w)
    error_loss += (outputs - y_real[CAU_NUM - 1]) ** 2
# print(err)
print(error_loss / PERDECT_NUM)





