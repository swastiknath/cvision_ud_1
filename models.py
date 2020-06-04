'''
AUTHOR : SWASTIK NATH
UDACITY COMPUTER VISION NANODEGREE PROGRAM.
CNN NETWORK FOR FACIAL LANDMARK DETECTION.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## 3. The last layer output 136 values, 2 for each of the 68 keypoint (x, y) pair
        
        #-- Convolution 1 :  image size 224 x 224 :
        # output size: ( w -f )/s + 1 = 224 - 5 /1 + 1 = 220 -> (64, 110, 110)
        self.conv1 = nn.Conv2d(1, 64, 5)
        I.xavier_uniform_(self.conv1.weight)
        self.conv1.bias.data.fill_(0.01)
        self.pool = nn.MaxPool2d(2, 2)
        
        #-- Convoulutionn 2: (64, 110, 110)
        # output size :  (w -f )/s +1 = 110 - 3 +1 = 108 -> (32, 54, 54)
        self.conv2 = nn.Conv2d(64, 32, 3)
        I.xavier_uniform_(self.conv2.weight)
        self.conv2.bias.data.fill_(0.01)
        
        #-- Convolution 3: (32, 54, 54)
        # output size : (W-f)/s + 1 = 54 - 5 +1 = 50 -> (32, 25, 25)
        self.conv3 = nn.Conv2d(32, 16, 5)
        I.xavier_uniform_(self.conv3.weight)
        self.conv3.bias.data.fill_(0.01)        
        
        #-- Fully connected Layers
        # output size -> 16 outputs * 25x25 activated maps -> 10,000
        self.fc1 = nn.Linear(16*25*25, 500)
        I.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 136)
        I.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
       
        
        
    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(self.dropout(x))
        
        return x
