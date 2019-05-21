import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models.alexnet as alexnet
from dataloader import CustomDataset
from torch.utils.data import DataLoader

image_path = './database/'
batch_size = 4

# Loading Data
print('Loading Data: ...')
trainset = CustomDataset(image_path=image_path, normalise=True, resize=(224, 224), train=True)
trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
print('Loading Data: DONE\n')

# Creating Network
net = alexnet()

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('Training: ...')
for epoch in range(2):  # loop over the dataset multiple times
    
    print('    ====== EPOCH ' + str(epoch+1) + ' ======')
    
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('    --> [%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Training: DONE')















