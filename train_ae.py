import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ae as ae

import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
 
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

data = np.array(pd.read_csv(sys.argv[1], sep='\t', header=None).T)
output_dir = str(sys.argv[2])
NUM_EPOCHS = int(sys.argv[3])
LEARNING_RATE = float(sys.argv[4])
INPUT_DIM = int(sys.argv[5])
HIDDEN_DIM = int(sys.argv[6])
LATENT_DIM = int(sys.argv[7])
details = str(LEARNING_RATE) + ' learning rate, ' + str(HIDDEN_DIM) + ' hidden dimensions, ' + str(LATENT_DIM) + ' latent dimensions'

print('checkpoint 1')

BATCH_SIZE = 128

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

net = ae.Autoencoder([INPUT_DIM, HIDDEN_DIM, LATENT_DIM])

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

sample = np.random.choice(len(data), size = int(len(data)*4/5), replace=False)
train = data[sample]
test = np.delete(data, sample, axis=0)
trainloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

print('checkpoint 2')

device = get_device()
net = net.to(device).float()

def train(net, trainloader, testloader, NUM_EPOCHS):
    train_loss = []
    test_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for point in trainloader:
            optimizer.zero_grad()
            point = point.to(device).float()
            loss = criterion(net(point), point)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss / len(trainloader))
        
        print('Epoch {} of {}, Train Loss: {:.5f}'.format(
            epoch+1, NUM_EPOCHS, loss))
        
        running_loss = 0.0
        for point in testloader:
            point = point.to(device).float()
            running_loss += criterion(net(point), point).item()
        test_loss.append(running_loss / len(testloader))
    
    return(train_loss,test_loss)

device = get_device()
net.to(device)

print('checkpoint 3')

train_loss,test_loss = train(net, trainloader, testloader, NUM_EPOCHS)
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)
ax1.plot(train_loss, label='training')
ax1.plot(test_loss, label='test')
plt.title("Autoencoder with "+details)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(output_dir+".png", bbox_inches='tight')

torch.save(net, 'results/models/bins/'+str(NUM_EPOCHS)+'_'+str(LEARNING_RATE)+'_'+str(HIDDEN_DIM)+'_'+str(LATENT_DIM)+'.model')
