import torch

torch.manual_seed(1)
X = torch.Tensor([[-1, 1, 2],[1, 1, 1]])
y = torch.Tensor([-1, 1, 1])
w = torch.Tensor([[0.1],[0.1]])
alpha = 1

for iter in range(100):
    tmp = torch.exp(torch.matmul(torch.transpose(w,0,1),X)*(-y))

    ##############################
    ## Use tmp to compute f and g. Instead of summing we average the result, i.e.,
    ## complete only inside torch.mean(...) and don't remove this function
    ## Dimensions: f (scalar); g (2)
    ##############################
    f = torch.mean()
    g = torch.mean(,1)

    print("Loss: %f; ||g||: %f" % (f, torch.norm(g)))
    g = g.view(-1,1)
    w = w - alpha*g

print(w)