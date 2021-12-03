
# 测试各种小函数用
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#
# rnn = nn.GRU(10, 20, 2)
# inputs = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# output, hn = rnn(inputs, h0)
#
# print(output.shape)
# print(hn.shape)

# 假设是时间步T1的输出
# T1 = torch.tensor([[1, 2, 3],
#         		[4, 5, 6],
#         		[7, 8, 9]])
# # 假设是时间步T2的输出
# T2 = torch.tensor([[10, 20, 30],
#         		[40, 50, 60],
#         		[70, 80, 90]])
#
# T3 = torch.tensor([[1, 2, 3],
#         		[4, 5, 6],
#         		[7, 8, 9],
#                    [7, 8, 9], ])
#
# T4 = torch.reshape(T3, (-1, 2, 3))
#
# T5 = T4.transpose(1, 2)
#
# print(T5[:, 0, 0])
#
# a = torch.stack((T1, T2), dim=0)
# b = torch.stack((T1, T2), dim=1)
# c = torch.stack((T1, T2), dim=2)
#
# cc = T1[:, 0:2]
# print(cc)
#
# print(torch.stack((T1, T2), dim=0).shape)
# print(torch.stack((T1, T2), dim=1).shape)
# print(torch.stack((T1, T2), dim=2).shape)
# # print(torch.stack((T1, T2), dim=3).shape)
# # outputs:
# torch.Size([2, 3, 3])
# torch.Size([3, 2, 3])
# torch.Size([3, 3, 2])

# a = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
# b = a.reshape(3, 2, 2)
# c = 1
#
# bsda = a.size()[0]
# print(bsda)

# a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
# b = a.permute(0, 2, 1)
# print(b)
# print(b.shape)

# nnm = torch.nn.ConvTranspose1d(5, 4, 3)
# inputs = torch.rand(32, 5, 100)
# outputs = nnm(inputs)
# print(outputs.shape)


# lossfn = nn.MSELoss()
# inputs = torch.randn(1, 2, 1, requires_grad=True)
# target = torch.randn(1, 2, 1)
#
#
# output = lossfn(inputs, target)
# output.backward()



# inputs = torch.rand(2, 2, 1, requires_grad=True)
# target = torch.rand(2, 2, 1)
# lossfn = F.binary_cross_entropy(inputs, target, reduction='none')
# lossfn2 = F.binary_cross_entropy(inputs, target)
# print(lossfn)
# print(lossfn2)

# BATCH_SIZE = 2
# y_pred = torch.rand(BATCH_SIZE, 2, 1, requires_grad=True)
# y_true = torch.rand(BATCH_SIZE, 2, 1)
# loss_val = torch.zeros(BATCH_SIZE)
# for jk in range(BATCH_SIZE):
#     loss_val[jk] = torch.mean(2 * (y_true[jk, :, :] - 0.5) *
#                               F.binary_cross_entropy(y_pred[jk, :, :],
#                                                      y_true[jk, :, :], reduction='none'))
# los = torch.sum(loss_val)
# print(los)



# def my_accuracy(y_true, y_pred):
#     acc = 0
#     for i in range(BATCH_SIZE):
#         for j in range(TIME_SEQ):
#             acc += 2 * torch.abs(y_true[i, j, :] - 0.5) * torch.equal(y_true[i, j, :], torch.round(y_pred[i, j, :]))
#     acc = acc/BATCH_SIZE/TIME_SEQ
#     return acc

# y_pred = torch.tensor([[1, 2], [1, 3]])
# y_true = torch.tensor([[1, 2], [1, 2]])
#
# c = torch.equal(y_true, y_pred)
# print(c)


fid = open('../basis_net/predict_dat/dat4/vadtest3.pcm', "rb")
tmp_pcm = fid.read()
fid.close()
pcm = np.frombuffer(tmp_pcm, np.int16, count=-1)
pcm = pcm[:512*100]
pcm_tensor = torch.from_numpy(pcm)
# pcm_tensor
# fty = torch.stft(pcm_tensor,  n_fft=512, hop_length=100, win_length = 200)
# rea = fty[:, :, 0]  #实部
# imag = fty[:, :, 1] #虚部
# mag = torch.abs(torch.sqrt(torch.pow(rea, 2) + torch.pow(imag, 2)))
# a = 1

wav_data1 = torch.tensor(np.arange(0, 100, 0.1))
a1 = torch.stft(wav_data1, n_fft=512, hop_length=100, win_length = 200)
aa1 = torch.istft(a1, n_fft=512, hop_length=100, win_length = 200)
# rea = a1[:, :, 0]#实部
# imag = a1[:, :, 1]#虚部
# mag = torch.abs(torch.sqrt(torch.pow(rea, 2) + torch.pow(imag, 2)))
# pha = torch.atan2(imag.data, rea.data)


