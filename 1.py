import torch 

a = [torch.rand(1, 1) for _ in range(1)]
b = [torch.rand(1, 1) for _ in range(1)]

res = torch._foreach_add(a, 1)
print(res)