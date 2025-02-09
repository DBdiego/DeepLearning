import time
import os
import re
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


'''
n_c_layers:   int, number of convolution layers
dim1:         int array with number of output channels for each convolution
kernel_conv:  int array with kernel size for each convolution layer
stride_conv:  int array with stride for each convolution layer
kernel_pool:  int array with kernel size for each pool layer
stride_pool:  int array with stride for each pool layer
n_l_layers:   int, number of linear layers
dim2:         int array with number of nodes for every linear layer
'''
class Net(nn.Module):
    def __init__(self, n_c_layers, dim1, kernel_conv, stride_conv, kernel_pool, stride_pool, n_l_layers, dim2):
        # first input channel is 3, followed by dim1:
        dim1.insert(0, 3)
        self.dim1 = dim1

        final_dim = 50 # decreases in size with each convolution and pooling layer
        super(Net, self).__init__()

        # Defining convolution layers
        self.n_c_layers = n_c_layers
        for i in range(n_c_layers):
            pad = int(kernel_conv[i]/2) # padding such that convolution shits out roughly same size
            setattr(self, f"conv{i}", nn.Conv2d(in_channels=self.dim1[i], out_channels=self.dim1[i+1],
                                                kernel_size=kernel_conv[i], stride=stride_conv[i], padding=pad))
            
            #(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            self.pool = nn.MaxPool2d(kernel_pool[i], stride_pool[i]) # (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

            # Updating changing of image size
            final_dim = int((final_dim - kernel_conv[i] + 2 * pad) / stride_conv[i] + 1)
            final_dim = int((final_dim - kernel_pool[i] + 2 * 0  ) / stride_pool[i] + 1)
            
        self.final_dim = final_dim
        
        # adding first layer (image to nodes):
        dim2.insert(0, self.dim1[-1]*self.final_dim*self.final_dim)
        
        # Defining network linear layers
        self.n_l_layers = n_l_layers
        for i in range(n_l_layers):
            setattr(self, f"fc{i}", nn.Linear(int(dim2[i]), int(dim2[i+1])))


    def forward(self, x):
        # Convolution layers:
        for i in range(self.n_c_layers):
            x=self.pool(F.relu(getattr(self, f"conv{i}")(x)))

        x = x.view(-1, self.dim1[-1] * self.final_dim * self.final_dim) # image to tensor structure ?

        # Neural net layers:
        for i in range(self.n_l_layers):
            if i != self.n_l_layers-2:
                x = F.relu(getattr(self, f"fc{i}")(x))
            else:
                x = getattr(self, f"fc{i}")(x)
        return x


'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.final_dim = 50
        self.final_dim = int((self.final_dim - 3 + 2 * 1) / 1 + 1)
        self.final_dim = int((self.final_dim - 2 + 2 * 0) / 2 + 1)
        self.final_dim = int((self.final_dim - 3 + 2 * 1) / 1 + 1)
        self.final_dim = int((self.final_dim - 2 + 2 * 0) / 2 + 1)
        self.conv1 = nn.Conv2d(3, 6, 3,padding=1)  # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = nn.Conv2d(6, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * self.final_dim * self.final_dim,
                             120)  # (in_features, out_features, bias=True), 7x7 is image size after conv & pooling...
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)  # 10 outputs (classes above)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.final_dim * self.final_dim)  # image to tensor structure ?
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''
