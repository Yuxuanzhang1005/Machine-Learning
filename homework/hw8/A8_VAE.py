from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
torch.manual_seed(2)


train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,
                                            transform=transforms.ToTensor()),batch_size=128, shuffle=True, num_workers=0)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2_1 = nn.Linear(400, 20)
        self.fc2_2 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_1(h1), self.fc2_2(h1)

    def reparameterize(self, mu, logvar):
        ##############################
        ## implement the reparameterize function
        ##############################
        std = 
        eps = 
        return 

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def criterion(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduce=True, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

num_epochs = 30
batch_size = 128

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = criterion(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('===== Epoch: {} Average loss: {:.4f} ======'.format(
          epoch, train_loss / len(train_loader.dataset)))


if __name__ == "__main__":
    for epoch in range(num_epochs):
        train(epoch)
        with torch.no_grad():
            sample = torch.randn(int(batch_size/2), 20).to(device)
            sample = model.decode(sample)
            save_image(sample.view(int(batch_size/2), 1, 28, 28),
                       'results/sampled_output_' + str(epoch) + '.png')    
