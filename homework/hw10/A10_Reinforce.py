import torch
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

pGT = torch.Tensor([1./12, 2./12, 3./12, 3./12, 2./12, 1./12])
y = torch.from_numpy(np.random.choice(list(range(6)), size=1000, p=pGT.numpy())).type(torch.int64).view(-1, 1)
delta = torch.zeros(y.numel(),6).scatter(1,y,torch.ones_like(y).float())

#maximum likelihood given dataset y encoded in delta
def MaxLik(delta):
    alpha = 1
    theta = torch.randn(6)
    for iter in range(100):
        p_theta = torch.nn.Softmax(dim=0)(theta)
        g = torch.mean(p_theta-delta,0)
        theta = theta - alpha*g
        print("Diff: %f" % torch.norm(p_theta - pGT))
    
    return theta

theta = MaxLik(delta)

#reinforce with reward R
def Reinforce(R, theta=None):
    alpha = 1
    if theta is None:
        theta = torch.randn(6)
    for iter in range(10000):
        #current distribution
        p_theta = torch.nn.Softmax(dim=0)(theta)

        #sample from current distribution and compute reward
        ##############################
        ## Sample from p_theta, find the assignment delta and compute the reward
        ## for each sample
        ## Dimensions: cPT (6); y (1000x1 -> 1000x1); delta (1000x6); curReward (1000x1)
        ############################## 
        y = 
        delta = 
        curReward = 

        #compute gradient and update
        g = torch.mean(curReward*(delta - p_theta),0)
        theta = theta + alpha*g
        print("Diff: %f" % torch.norm(p_theta - pGT))
        print(p_theta)

R = pGT
Reinforce(R, theta)
    
