import time
import os
import re

import torch.nn as nn
import torch.nn.functional as F


# Parameters: [convolution, maxpooling]
class Net(nn.Module):
    def __init__(self):
        kernel_size = [3, 2]
        padding = [1, 0]
        stride = [1, 2]
        img_x = 50
        dim1 = ((img_x - kernel_size[0] + 2 * padding[0]) / stride[0] + 1)
        dim2 = ((dim1 - kernel_size[1] + 2 * padding[1]) / stride[1] + 1)
        dim3 = ((dim2 - kernel_size[0] + 2 * padding[0]) / stride[0] + 1)
        dim4 = ((dim3 - kernel_size[1] + 2 * padding[1]) / stride[1] + 1)
        self.final_dim = int(dim4)
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size[0], padding=padding[0]) # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.pool = nn.MaxPool2d(2, 2) # (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.conv2 = nn.Conv2d(6, 16, kernel_size[0], padding=padding[0])
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * self.final_dim * self.final_dim, 120) # (in_features, out_features, bias=True), 7x7 is image size after conv & pooling...
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 40)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.final_dim * self.final_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x