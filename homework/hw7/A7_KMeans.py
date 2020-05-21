import torch
import numpy as np
import matplotlib.pyplot as plt

N = 10
std = 0.5
torch.manual_seed(1)
x = torch.cat((std*torch.randn(2,N)+torch.Tensor([[2],[-2]]), std*torch.randn(2,N)+torch.Tensor([[-2],[2]])),1)

def Plot(c):
    plt.plot(x[0,:N].numpy(), x[1,:N].numpy(), 'ro')
    plt.plot(x[0,N:].numpy(), x[1,N:].numpy(), 'bo')
    l = plt.plot(c[0,:].numpy(), c[1,:].numpy(), 'kx')
    plt.setp(l, markersize=10)
    plt.show()

c = torch.Tensor([[2, -2],[2, -2]])
ctmp = c.transpose(0,1).view(2,2,1)

Plot(c)

for iter in range(10):
    ##############################
    ## compute the distance between points and cluster centers
    ## Dimensions: dist (2x20)
    ##############################
    dist = 

    val,assign = dist.min(0)
    print("Cost: %f" % torch.sum(val))
    for k in range(ctmp.size()[0]):
        mn = torch.mean(x[:,assign==k],1)
        ctmp[k,:,:] = mn.view(-1,1)
    
    Plot(c)

print(ctmp)
