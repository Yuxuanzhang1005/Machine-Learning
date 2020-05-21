import torch
import torch.optim as optim
import torch.nn as nn

torch.manual_seed(1)
alpha = 1
C = 0

##############################
## encode the dataset to fit the one specified in HW4.pdf (note that bias
## is part of the network now)
## Dimensions: X (2x3); y (3)
##############################
X = torch.Tensor([[, , ],[, , ]])
y = torch.LongTensor([, , ])

class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.fc1 = nn.Linear(2,3, bias=True)
    
    def forward(self, X):
        return self.fc1(X)

net = ShallowNet()
print(net)

print(net(torch.transpose(X,0,1)).squeeze())

optimizer = optim.SGD(net.parameters(), lr=alpha, weight_decay=C)
optimizer.zero_grad()

criterion = nn.CrossEntropyLoss()

for iter in range(10000):
    netOutput = net(torch.transpose(X,0,1))

    ##############################
    ## provide the arguments for the criterion function
    ## Dimensions: loss (scalar)
    ##############################    
    loss = criterion(, )

    loss.backward()
    gn = 0
    for f in net.parameters():
        gn = gn + torch.norm(f.grad)
    print("Loss: %f; ||g||: %f" % (loss, gn))

    ##############################
    ## Use two functions within the optimizer instance to perform the update step
    ##############################    
    optimizer.
    optimizer.

for f in net.parameters():
    print(f)

print(nn.Softmax(dim=1)(net(torch.transpose(X, 0, 1))))

