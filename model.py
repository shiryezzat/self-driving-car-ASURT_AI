import torch
from torch import nn

## Put your Architecture here

## Here is a sample and this one not supposed to work so delete it and put your own
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam


# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(273504, 2)

        self.optimizer = optim.Adam(self.parameters(), lr = 0.001)
        self.criterion = nn.MSELoss()
        
    def forward(self, input):
        input = input.reshape((-1, 3, 160, 320))
        
        #print(input.shape)
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = self.flat(output)
        output = self.fc1(output)

        return output
