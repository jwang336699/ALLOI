import os
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, dims):
        super(Autoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=dims[0], out_features=dims[1])
        self.enc2 = nn.Linear(in_features=dims[1], out_features=dims[2])
        # decoder 
        self.dec1 = nn.Linear(in_features=dims[2], out_features=dims[1])
        self.dec2 = nn.Linear(in_features=dims[1], out_features=dims[0])

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        return x

    def decode(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        return x

    def forward(self, x):
        return self.decode(self.encode(x))
