import h5py
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from model import Vad2Net


# Hyper Parameters
TIME_STEP = 40              # rnn time step  使语音长度，1帧10ms
TIME_SEQ = TIME_STEP
# OUT_H_SIZE = 15           # rnn hidden size
LR = 0.001                  # learning rate
INPUT_CH = 42               # feature 维度， 图片宽度， 也有可能直接用
NDIM = INPUT_CH

BATCH_SIZE = 25
EPOCH_NUM = 100             # 训练次数
OUTSIZE = 1                 # VAD最终输出1


def myloss(y_true, y_pred):
    # 这里 y_pred.shape = (BATCH_SIZE, TIME_SEQ, 1), loss time_seq需要平均化的，但是batch是要累加的
    loss_val = torch.zeros(BATCH_SIZE)
    for jk in range(BATCH_SIZE):
        loss_val[jk] = torch.mean(2*torch.abs(y_true[jk, :] - 0.5) *
                                  F.binary_cross_entropy(y_pred[jk, :],
                                                         y_true[jk, :], reduction='none'))
    los = torch.sum(loss_val)
    return los


def my_accuracy(y_true, y_pred):
    acc = 0
    for i in range(BATCH_SIZE):
        acc += 2 * torch.abs(y_true[i, :] - 0.5) * torch.equal(y_true[i, :], torch.round(y_pred[i, :]))
    acc = acc/BATCH_SIZE
    return acc


vadnet = Vad2Net()
print(vadnet)


# 数据集导入
print('Loading data...')
with h5py.File('../train_dat/training.h5', 'r') as hf:
    all_data = hf['data'][:]
print('done.')


# window_size = 2000
nb_sequences = len(all_data)
x_t = all_data[:nb_sequences, :42]
vad_t = np.copy(all_data[:nb_sequences, 86:87])
# ## 重新处理数据使其满足网络要求，最后预留10%作为测试集
tran_len = np.int64(nb_sequences*0.9)

x_train = x_t[:tran_len, :]
y_train = vad_t[:tran_len, :]

x_test = x_t[tran_len+1:, :]
y_test = vad_t[tran_len+1:, :]

trains_d = x_train.reshape(-1, BATCH_SIZE, TIME_SEQ, NDIM)
labels_l = y_train.reshape(-1, BATCH_SIZE, TIME_SEQ, OUTSIZE)

trains_d_tensor = torch.from_numpy(trains_d)
labels_l_tensor = torch.from_numpy(labels_l)

BLOCK = trains_d_tensor.size()[0]

optim = torch.optim.Adam(vadnet.parameters(), lr=LR)

for epoch in range(EPOCH_NUM):
    loss_epoch = 0.0
    acc_sum = 0.0
    NUM = BLOCK * BATCH_SIZE
    for i in range(BLOCK):
        trains_d_s = trains_d_tensor[i, :, :, :]
        labels_l_s = labels_l_tensor[i, :, TIME_STEP-1, :]
        trains_d_st = trains_d_s.permute(0, 2, 1)
        optim.zero_grad()
        outputs = vadnet(trains_d_st)                #
        # labels_l_y = labels_l_s[:, :, NDIM - OUT_SIZE:NDIM]
        # loss = torch.mean(2 * (labels - 0.5) * F.binary_cross_entropy(outputs, labels, reduction='none'))
        loss = myloss(labels_l_s, outputs)
        y_pre_l = torch.round(outputs)
        loss.backward()
        optim.step()
        acc_b = my_accuracy(labels_l_s, outputs)
        acc_sum += acc_b
        loss_epoch += loss.item()
        # print("acc:{},  loss:{}".format(acc_b, loss))
    print("epoch:{},  loss:{}, acc:{}".format(epoch, loss_epoch/NUM, acc_sum/BLOCK))


save_path = 'Vad2Net.pth'
torch.save(vadnet.state_dict(), save_path)       # 将训练好的模型保存下来
params = vadnet.state_dict()                     # 获得模型的原始状态以及参数。



