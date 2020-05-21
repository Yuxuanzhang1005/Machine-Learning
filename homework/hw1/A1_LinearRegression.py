import torch

x = torch.Tensor([[1],[2]])
y = torch.Tensor([[1],[1]])
print(x.size())

X = torch.cat((x, torch.ones_like(x)),1)
print(X)
print(X[1,0])
print(torch.matmul(X, y))

# Solution 1
##############################
## Fill in the arguments
##############################
res1 = torch.gels(,)
print("Solution 1:")
print(res1[0])

# Solution 2
print(torch.matmul(torch.transpose(X, 0, 1),X))
print(torch.matmul(torch.transpose(X, 0, 1),y))

##############################
## How to compute l and r?
## Dimensions: l (2x2); r (2x1)
##############################
l = 
r = 
res2 = torch.gesv(r,l)
print("Solution 2:")
print(res2[0])

# Solution 3
##############################
## What is l and r?
## Dimensions: l (2x2); r (2x1)
##############################
l = 
r = 
res3 = torch.matmul(torch.inverse(l),r)
print("Solution 3:")
print(res3)

