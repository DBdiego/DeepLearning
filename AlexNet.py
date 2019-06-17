import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models.alexnet as alexnet
import time
import numpy as np
from dataloader import CustomDataset
from torch.utils.data import DataLoader, random_split

image_path = './database/'

# Parameters Geoffrey look here for parameter!!
batch_size   = 40
maxtraintime = 20*60         # seconds note that testing time is not included this takes +- 5 minutes for AlexNet at my laptop, which is almost the same as 1 epoch
lengths      = [10000,10778] # training data, test data
convergence  = 0.001         # Not sure if this is a good value (smaller change than 0.1%)
minepoch     = 6             # should be 6 or higher, it can have less epochs in results if the maxtraintime is exceeded.

NORMALIZE = True
IMAGE_PATH = 'database/'
no_classes = [5,8,10,20,40]
imgs_classes = [3299,4296,5434,10521,20778] # number of images for number of classes above
CLASSES_INDEX = 0 # NOTE: have to change line 18 in batch_population as well
RATIO_TRAINING = 0.3
RATIO_DATA = 1
MAX_DATA = RATIO_DATA * 2 * imgs_classes[CLASSES_INDEX]#41556



# Loading Data
print('Loading Data: ...')
dataset = CustomDataset(image_path=IMAGE_PATH, normalise=NORMALIZE, maxx=MAX_DATA, tot_imgs=imgs_classes[CLASSES_INDEX])
print('Importing data: DONE\n')

I = int(RATIO_TRAINING * len(dataset))
lengths = [len(dataset) - I, I]  # train data and test data
train_dataset, test_dataset = random_split(dataset, lengths)
trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
testloader  = DataLoader(test_dataset, shuffle=False, num_workers=1)
print('Loading Data: DONE\n')

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
# Creating Network
net = alexnet()
net = net.to(device)

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('Training: ...')

epoch = 0
losslst = []
starttime = time.time()
traintime = time.time() - starttime

#for epoch in range(2):  # loop over the dataset multiple times

while traintime < maxtraintime:
    epoch = epoch + 1   
#    pytorch_total_params = sum(p.numel() for p in net.parameters())
#    print(pytorch_total_params)
    running_loss_epoch = 0.0 
    running_loss = 0.0
    
    if epoch > minepoch:  # minimum number of epochs
        rule = abs(np.mean(np.diff(losslst[-5:])))/losslst[-5:][0]
        if rule < convergence:
            realtime = time.time() - starttime
            break
            
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        if traintime > maxtraintime:
            print( 'traintime:', traintime)
            break
        
        inputs, labels = data
        
        inputs = inputs.to(device)#cuda()
        labels = labels.to(device)#cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            running_loss_epoch += running_loss
            running_loss = 0.0
            
        if epoch == 1 and i == 0:
            batchtime = time.time() - starttime
            
        traintime = time.time() - starttime + batchtime # + batchtime estimates the time for the next batch
        realtime = time.time() - starttime

        
    losslst.append(running_loss_epoch)
    print(f'\t AlexNet: epoch {epoch} loss:', round(running_loss_epoch, 5), f'on {i} minibatches {(running_loss_epoch/i)/batch_size}')
    
    
print('Training: DONE')
# --------------------------------
# Testing:
# Whole test data set
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Accuracy of the network on the 10000 test images: %d %%' % (
    accuracy))

print('Training time:', realtime)


#important variables are: accuracy, realtime, losslst (list of loss after each epoch)







