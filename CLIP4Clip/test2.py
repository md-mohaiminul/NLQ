import torch

x = torch.tensor([[1,2,3], [4,5,6]])

y = torch.tensor([2,2,2])

print(torch.matmul(x,y))

print(x[:,1])