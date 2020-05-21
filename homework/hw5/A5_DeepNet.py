import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import struct

torch.manual_seed(1)

class OurDataset(Dataset):
    def __init__(self, fnData, fnLabels, transform=None):
        self.transform = transform
        self.LoadData(fnData)
        self.LoadLabels(fnLabels)
        assert self.l.size()[0]==self.d.size()[0]
    
    def LoadLabels(self, fnLabels):
        fid = open(fnLabels,'rb')
        head = fid.read(8)
        data = fid.read()
        fid.close()

        res = struct.unpack('>ii',head)
        data1 = struct.unpack(">"+"B"*res[1],data)
        self.l = torch.LongTensor(data1)

    def LoadData(self, fnData):
        fid = open(fnData,'rb')
        head = fid.read(16)
        data = fid.read()
        fid.close()

        res = struct.unpack(">iiii", head)
        data1 = struct.iter_unpack(">"+"B"*784,data)

        self.d = torch.zeros(res[1],1,res[2],res[3])
        for idx,k in enumerate(data1):
            tmp = torch.Tensor(k)
            tmp = tmp.view(1,res[2],res[3])
            if self.transform:
                tmp = self.transform(tmp)
            self.d[idx,:,:,:] = tmp
        
    def __len__(self):
        return self.d.size()[0]
    def __getitem__(self, idx):
        return (self.d[idx,:,:], self.l[idx])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ##############################
        ## declare the layers of the network which have parameters
        ##############################
        self.conv1 = 
        self.conv2 = 
        self.fc1 = 
        self.fc2 = 

    def forward(self, x):
        ##############################
        ## combine the layers; don't forget the relu and pooling operations
        ##############################
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = 
        x = 
        x = x.view(, )
        x = 
        return 

testData = OurDataset('MNIST/t10k-images-idx3-ubyte','MNIST/t10k-labels-idx1-ubyte',transform=transforms.Compose([
                           transforms.Normalize((255*0.1307,), (255*0.3081,))
                       ]))
trainData = OurDataset('MNIST/train-images-idx3-ubyte','MNIST/train-labels-idx1-ubyte',transform=transforms.Compose([
                           transforms.Normalize((255*0.1307,), (255*0.3081,))
                       ]))
print(testData.__len__())
print(trainData.__len__())

trainLoader = DataLoader(trainData, batch_size=128, shuffle=True, num_workers=0)
testLoader = DataLoader(testData, batch_size=128, shuffle=False, num_workers=0)

net = Net()

numparams = 0
for f in net.parameters():
    print(f.size())
    numparams += f.numel()

optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=0)
optimizer.zero_grad()

criterion = nn.CrossEntropyLoss()

def test(net, testLoader):
    net.eval()
    correct = 0
    with torch.no_grad():
        for (data,target) in testLoader:
            output = net(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
        print("Test Accuracy: %f" % (100.*correct/len(testLoader.dataset)))

test(net, testLoader)

for epoch in range(10):
    net.train()
    for batch_idx, (data, target) in enumerate(trainLoader):
        pred = net(data)
        loss = criterion(pred, target)
        loss.backward()
        gn = 0
        for f in net.parameters():
            gn = gn + torch.norm(f.grad)
        #print("E: %d; B: %d; Loss: %f; ||g||: %f" % (epoch, batch_idx, loss, gn))
        optimizer.step()
        optimizer.zero_grad()
    
    test(net, testLoader)

