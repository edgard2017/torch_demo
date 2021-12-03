

import numpy as np
import torch
import torch.nn as nn
from model import Vad1Net
import matplotlib.pyplot as plt


# 读取音频pcm
fid = open('../predict_dat/dat4/vadtest3.pcm', "rb")
tmp_pcm = fid.read()
fid.close()
pcm = np.frombuffer(tmp_pcm, np.int16, count=-1)

# 读取特征
fid = open('../predict_dat/dat4/feature.dat', "rb")
tmp = fid.read()
fid.close()
feature = np.frombuffer(tmp, np.float32, count=-1)
feature = feature.reshape(-1, 42)
feature = torch.from_numpy(feature)
feature = torch.unsqueeze(feature, 0)
feature.requires_grad = False

vadnet = Vad1Net()
print(vadnet)
vadnet.load_state_dict(torch.load('Vad1Net.pth'))  # 导入已经训练好的网络参数

with torch.no_grad():
    vad_labels = vadnet(feature)

vad_labels = torch.squeeze(vad_labels)


vad_label_np = vad_labels.numpy()
vad_label_np2 = np.tile(vad_label_np, (480, 1))
vad_label_np2 = vad_label_np2.T
vad_label_np3 = vad_label_np2.reshape(1, -1)
vad_label_np3 = np.squeeze(vad_label_np3)

plt.subplot(211)
plt.figure(1)
plt.plot(vad_label_np3)
plt.xlim(0, vad_label_np3.size)

plt.subplot(212)
plt.figure(1)
plt.plot(pcm)
plt.xlim(0, pcm.size)
plt.show()



