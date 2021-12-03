import h5py
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from model import Vad1Net


EPOCH_NUM = 100
BATCH_SIZE = 32
OUTSIZE = 1
TIME_SEQ = 1000
NDIM = 42


def myloss(y_true, y_pred):
    # 这里 y_pred.shape = (BATCH_SIZE, TIME_SEQ, 1), loss time_seq需要平均化的，但是batch是要累加的
    loss_val = torch.zeros(BATCH_SIZE)
    for jk in range(BATCH_SIZE):
        loss_val[jk] = torch.mean(2*torch.abs(y_true[jk, :, :] - 0.5) *
                                  F.binary_cross_entropy(y_pred[jk, :, :],
                                                         y_true[jk, :, :], reduction='none'))
    los = torch.sum(loss_val)
    return los


def my_accuracy(y_true, y_pred):
    acc = 0
    for i in range(BATCH_SIZE):
        for j in range(TIME_SEQ):
            acc += 2 * torch.abs(y_true[i, j, :] - 0.5) * torch.equal(y_true[i, j, :], torch.round(y_pred[i, j, :]))
    acc = acc/BATCH_SIZE/TIME_SEQ
    return acc


vadnet = Vad1Net()
print(vadnet)


# 数据集导入
print('Loading data...')
with h5py.File('../train_dat/training_28.h5', 'r') as hf:
    all_data = hf['data'][:]
print('done.')


# window_size = 2000
nb_sequences = len(all_data)
x_t = all_data[:nb_sequences, :42]
vad_t = np.copy(all_data[:nb_sequences, 86:87])
# ## 重新处理数据使其满足网络要求，最后预留10%作为测试集
tran_len = np.int64(nb_sequences)

x_train = x_t[:tran_len, :]
y_train = vad_t[:tran_len, :]

x_test = x_t[tran_len+1:, :]
y_test = vad_t[tran_len+1:, :]

trains_d = x_train.reshape(-1, BATCH_SIZE, TIME_SEQ, NDIM)
labels_l = y_train.reshape(-1, BATCH_SIZE, TIME_SEQ, OUTSIZE)

trains_d_tensor = torch.from_numpy(trains_d)
labels_l_tensor = torch.from_numpy(labels_l)

BLOCK = trains_d_tensor.size()[0]

optim = torch.optim.Adam(vadnet.parameters(), lr=0.0005)

for epoch in range(EPOCH_NUM):
    loss_epoch = 0.0
    acc_sum = 0
    acc_b = 0
    NUM = BLOCK * BATCH_SIZE
    for i in range(BLOCK):
        trains_d_s = trains_d_tensor[i, :, :, :]
        labels_l_s = labels_l_tensor[i, :, :, :]
        optim.zero_grad()
        outputs = vadnet(trains_d_s)                #
        # labels_l_y = labels_l_s[:, :, NDIM - OUT_SIZE:NDIM]
        # loss = torch.mean(2 * (labels - 0.5) * F.binary_cross_entropy(outputs, labels, reduction='none'))
        loss = myloss(labels_l_s, outputs)
        loss.backward()
        optim.step()
        if epoch >= (EPOCH_NUM-5):
            acc_b = my_accuracy(labels_l_s, outputs)
        acc_sum += acc_b
        loss_epoch += loss.item()
        # print("acc:{},  loss:{}".format(acc_b, loss))
    print("epoch:{},  loss:{}, acc:{}".format(epoch, loss_epoch/NUM, acc_sum/BLOCK))


save_path = 'Vad1Net_28_1.pth'
torch.save(vadnet.state_dict(), save_path)       # 将训练好的模型保存下来
params = vadnet.state_dict()                     # 获得模型的原始状态以及参数。



