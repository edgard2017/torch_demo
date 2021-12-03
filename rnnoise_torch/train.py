
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import h5py
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from model import VadNet
import torch.optim as optim
import torchvision.transforms as transforms


def VadLoss(y_true, y_pred):
    # loss_mean = torch.tensor([1], dtype=float)
    # loss_mean = 2
    loss_val = torch.zeros(BATCH_SIZE)
    for i in range(BATCH_SIZE):
        loss_val[i] = torch.mean(2*(y_true[i, :, :] - 0.5) *
                                 F.binary_cross_entropy(y_pred[i, :, :],
                                                        y_true[i, :, :], reduction='none'))
    # loss_val.requires_grad = True
    return loss_val


# Hyper Parameters
TIME_STEP = 40              # rnn time step  使语音长度，1帧10ms
# OUT_H_SIZE = 15           # rnn hidden size
LR = 0.001                  # learning rate
INPUT_CH = 42               # feature 维度， 图片宽度， 也有可能直接用
OUTPUT_CH1 = 6               # 卷积核的个数，相当于深度，也就是2D中第二个参数
OUTPUT_CH2 = 6               # 卷积核的个数，相当于深度，也就是2D中第二个参数
K1_SIZE = 3                 # 卷积核的尺寸
K2_SIZE = 2                   #
BATCH_SIZE = 10             # 训练一次块，每一个字自己训练
# TEST_BATCH_SIZE = 1000      # 测试batch

# Net information
vadnet = VadNet()
print(VadNet)
# Input = torch.randn(BATCH_SIZE, INPUT_CH, TIME_STEP)
# output = vadnet(Input)
# print(output.shape)
# m = nn.Conv1d(16, 33, 3, stride=2)
# input = torch.randn(20, 16, 50)
# output = m(input)

# 数据集导入
print('Loading data...')
with h5py.File('training.h5', 'r') as hf:
    all_data = hf['data'][:]
print('done.')

# window_size = 2000
nb_sequences = len(all_data)
x_t = all_data[:nb_sequences, :42]
vad_t = np.copy(all_data[:nb_sequences, 86:87])
# vad_t = vad_t*2
#
# ## 重新处理数据使其满足网络要求，最后预留10%作为测试集
tran_len = np.int64(nb_sequences*0.9)

x_train = x_t[:tran_len, :]
y_train = vad_t[:tran_len, :]

x_test = x_t[tran_len+1:, :]
y_test = vad_t[tran_len+1:, :]

#
# loss函数使用常规的交叉熵
# loss_function = VadLoss()
# 优化器用adam，learning_rate = 0.0005
optimizer = torch.optim.Adam(vadnet.parameters(), lr=0.0005)

for epoch in range(100):
    running_loss = 0.0
    # 全部训练数据需要跑多少个循环
    for step in range(10000):
        inputs = torch.from_numpy(x_train[step*10:step*10+40, :].T)       # 这里做了下转置
        labels = torch.from_numpy(y_train[step*10+40-1])
        inputs = torch.unsqueeze(inputs, 0)
        # labels = labels.t().long()
        labels = torch.unsqueeze(labels, 0)
        # target = torch.empty(1, dtype=torch.long).random_(5)
        # target.item = labels.item()
        # zero the parameter gradients
        # if step % 10 == 9:
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = vadnet(inputs)
        # print("outputs = ", outputs.shape)
        # print("labels = ", labels.shape)
        # outputs = outputs.t().long()
        loss = torch.mean(2 * (labels - 0.5) * F.binary_cross_entropy(outputs, labels, reduction='none'))
        # print("loss = ", loss)
        # print("out = ", outputs)
        # print("lab = ", labels)
        # loss = VadLoss(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        # if step % 500 == 499:  # print every 500 mini-batches  [batch, channel, height, width]
        #     with torch.no_grad():  # 对于with不太理解
        #         inputs_t
        #         outputs = vadnet(val_image)  # [batch, 10]
        #         predict_y = torch.max(outputs, dim=1)[1]
        #         accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
        #
        #         print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
        #               (epoch + 1, step + 1, running_loss / 500, accuracy))
        #         running_loss = 0.0

    print("running_loss = ", running_loss)



