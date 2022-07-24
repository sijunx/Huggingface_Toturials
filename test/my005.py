import torch

a = torch.zeros(2, 3, 4)
print(a)

b = a[:, 0:1, :]
print(b.shape)
b = torch.squeeze(b)
print(b.shape)


