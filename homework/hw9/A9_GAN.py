import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import struct
import numpy as np
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2)

class OurDataset(Dataset):
    def __init__(self, fnData, dev, transform=None):
        self.transform = transform
        self.LoadData(fnData, dev)

    def LoadData(self, fnData, dev):
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

        self.d = self.d.to(dev)

    def __len__(self):
        return self.d.size()[0]
    def __getitem__(self, idx):
        return self.d[idx,:,:]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1) #out: 28 -> 14
        self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1) #out: 14 -> 7
        self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0) #out: 5 -> 5
        self.fc1 = nn.Linear(5*5*8, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 5*5*8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Gen(nn.Module):
    def __init__(self, zdim):
        super(Gen, self).__init__()
        self.firstDim = 16
        self.fc1 = nn.Linear(zdim, 4*4*self.firstDim)
        self.conv1 = nn.ConvTranspose2d( self.firstDim,  8, kernel_size=4, stride=2, padding=2, bias=False) #out: 6
        self.conv2 = nn.ConvTranspose2d( 8,  4, kernel_size=4, stride=2, padding=0, bias=False) #out: 14
        self.conv3 = nn.ConvTranspose2d( 4,  2, kernel_size=4, stride=2, padding=1, bias=False) #out: 28
        self.conv4 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.fc2 = nn.Linear(1, 784, bias=False)

    def forward(self, z):
        z = torch.nn.LeakyReLU(0.2)(self.fc1(z))
        z = z.view(z.size()[0],self.firstDim,4,4)
        z = torch.nn.LeakyReLU(0.2)(self.conv1(z))
        z = torch.nn.LeakyReLU(0.2)(self.conv2(z))
        z = torch.nn.LeakyReLU(0.2)(self.conv3(z))
        z = F.sigmoid(self.conv4(z))
        return z

def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        #print(torch.typename(m))
        torch.nn.init.xavier_uniform_(m.weight.data)
        #m.weight.data.fill_(0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if isinstance(m, torch.nn.Linear):
        #print(torch.typename(m))
        torch.nn.init.xavier_uniform_(m.weight.data)
        #m.weight.data.fill_(0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if isinstance(m, torch.nn.ConvTranspose2d):
        #print(torch.typename(m))
        torch.nn.init.xavier_uniform_(m.weight.data)
        #m.weight.data.fill_(0.01)
        if m.bias is not None:
            m.bias.data.fill_(0)

trainData = OurDataset('../MNIST/train-images-idx3-ubyte', dev, transform=transforms.Compose([
                           transforms.Normalize((255*0.,), (255.*1.0,))
                       ]))
print(trainData.__len__())

discIter = 1
genIter = 1
numEpoch = 250
batchSize = 256
zdim = 64

trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=0, drop_last=True)

disc = Net().to(dev)
gen = Gen(zdim).to(dev)

disc.apply(weights_init)
gen.apply(weights_init)

dopt = optim.Adam(disc.parameters(), lr=0.0002, weight_decay=0.0)
dopt.zero_grad()
gopt = optim.Adam(gen.parameters(), lr=0.0002, weight_decay=0.0)
gopt.zero_grad()

criterion = nn.BCEWithLogitsLoss()

target1 = torch.Tensor([1,]*batchSize + [0,]*batchSize).to(dev).view(-1,1)
target2 = torch.Tensor([1,]*batchSize).to(dev).view(-1,1)

for epoch in range(numEpoch):
    for batch_idx,data in enumerate(trainLoader):
        z = 2*torch.rand(data.size()[0], zdim, device=dev)-1
        xhat = gen(z)

        if batch_idx==0 and epoch==0:
            plt.imshow(data[0,0,:,:].detach().cpu().numpy())
            plt.savefig("goal.pdf")
            #plt.show()

        if batch_idx==0 and epoch%50==0:
            tmpimg = xhat[0:64,:,:,:].detach().cpu()
            save_image(tmpimg, "test_{0}.png".format(epoch), nrow=8, normalize=True)
            #plt.imshow(tmpimg[0,0,:,:].cpu().numpy())
            #plt.savefig("test_{0}.pdf".format(epoch))
            #plt.ion()
            #plt.show()
            #plt.pause(0.001)

        dopt.zero_grad()
        for k in range(discIter):
            logit = disc(torch.cat((data,xhat.detach()),0))
            ##############################
            ## implement the discriminator loss (-logD trick)
            ##############################
            loss = criterion(, )
            print("E: %d; B: %d; DLoss: %f" % (epoch,batch_idx,loss.item()))
            loss.backward()
            dopt.step()
            dopt.zero_grad()

        gopt.zero_grad()
        for k in range(genIter):
            xhat = gen(z)
            logit = disc(xhat)
            ##############################
            ## implement the generator loss (-logD trick)
            ##############################
            loss = criterion(, )
            loss.backward()
            print("E: %d; B: %d; GLoss: %f" % (epoch,batch_idx,loss.item()))
            gopt.step()
            gopt.zero_grad()