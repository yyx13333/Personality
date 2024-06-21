import torch


a = torch.zeros(3,3,3)


b = torch.randn(3,3,3)

c = a[2][2][:] = b[2][2][:]
print(a,b,c)