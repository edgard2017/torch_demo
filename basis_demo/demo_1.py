# 使用Seq写一个cnn的网络

import torch
from torch import Module, nn


class MyCnnNet(nn.Module):
    def __init__(self):
        super(MyCnnNet, self).__init__()
        model = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x




