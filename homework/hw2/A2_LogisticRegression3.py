import torch
import torch.optim as optim
import torch.nn as nn

torch.manual_seed(1)
X = torch.Tensor([[-1, 1, 2],[1, 1, 1]])
##############################
## modify the dataset so that it can be used here and is equivalent to the one 
## used in A2_LogisticRegression2.py and A2_LogisticRegression.py
## Dimensions: y (3)
##############################
y = torch.Tensor([, , ])

alpha = 1

class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.fc1 = nn.Linear(2,1, bias=False)
    
    def forward(self, X):
        return self.fc1(X)

net = ShallowNet()
print(net)

net.fc1.weight.data = torch.Tensor([[0.1, 0.1]])

print(net(torch.transpose(X,0,1)).squeeze())

optimizer = optim.SGD(net.parameters(), lr=alpha)
optimizer.zero_grad()

criterion = nn.BCEWithLogitsLoss()

for iter in range(100):
    netOutput = net(torch.transpose(X,0,1)).squeeze()

    ##############################
    ## provide the arguments for the criterion function
    ##############################
    loss = criterion(, )
    
    loss.backward()
    gn = 0
    for f in net.parameters():
        gn = gn + torch.norm(f.grad)
    print("Loss: %f; ||g||: %f" % (loss, gn))
    optimizer.step()
    optimizer.zero_grad()

for f in net.parameters():
    print(f)