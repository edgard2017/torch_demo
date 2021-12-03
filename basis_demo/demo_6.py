# Example of target with class indices
import torch
from torch import nn
import torch.nn.functional as F


# inputs = torch.randn((3, 2), requires_grad=True)
# target = torch.rand((3, 2), requires_grad=False)
inputs = [[1, 2, 3], [1, 2, 3]]
inputs = torch.tensor([[0.5]], dtype=float, requires_grad=True)

# inputs[0] = 1
# inputs.requires_grad = False
print(inputs)
target = torch.tensor([[1]], dtype=float)
loss = F.binary_cross_entropy(inputs, target, reduction='none')

loss1 = 2*(target-0.5)*loss
loss2 = torch.mean(loss1)


print(inputs.shape)

# loss = F.binary_cross_entropy(inputs, target)

print(loss)
print(loss1)
print(loss2)
# loss.backward()


