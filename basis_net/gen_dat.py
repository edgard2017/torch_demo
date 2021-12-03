
# 简单获取一点数据，来测试网络的训练效果
import torch


BATCH_SIZE = 32 #
TIME_SEQ = 100  # 时间序列
NDIM = 20   # fearture 维度
ELOOP = 10
BLOCK = 1000


DLEN = BATCH_SIZE * TIME_SEQ * NDIM * BLOCK
# DLEN = 30
FLEN = 5
x = torch.rand(DLEN)
w = torch.tensor([0.71123, 0.90514, 1.13258, 1.2236, 3.22766])
# w = torch.rand(FLEN)
# FLEN = len(w)
y = torch.zeros(DLEN)

for i in range(DLEN-FLEN):
    y[i+FLEN-1] = torch.sum(w*x[i:i+FLEN])



# x1 = torch.rand(DLEN)
# y1 = torch.zeros(DLEN)
# for i in range(DLEN-FLEN):
#     y1[i+FLEN-1] = torch.sum(w*x1[i:i+FLEN])

torch.save(x, 'train_dat/train_t.data')
torch.save(y, 'train_dat/train_l.data')
torch.save(w, 'train_dat/train_w.data')

# torch.save(x1, 'train_dat/test_t.data')
# torch.save(y1, 'train_dat/test_l.data')






