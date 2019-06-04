import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from cnn2 import Net
from dataloader import CustomDataset


# torch.manual_seed(0)


class CNN:

    def __init__(self, trainset, testset,
                 n_conv,
                 dim1,
                 kernel_conv,
                 stride_conv,
                 kernel_pool,
                 stride_pool,
                 n_layers,
                 dim2):

        # --------------------------------------
        # Parameters:
        maxtraintime = 20 * 60  # seconds, not sure if this is a good time. Note that testing time is not included, this is (often) slightly less than 1 epoch time.
        batch_size = 10
        lr = 0.001
        momentum = 0.9
        convergence = 0.001  # Not sure if this is a good value (smaller change than 0.1%)
        minepoch = 6  # should be 6 or higher, it can have less epochs in results if the maxtraintime is exceeded.

        # --------------------------------------
        # Data loading:
        trainloader = DataLoader(dataset=trainset,
                                 batch_size=batch_size,
                                 shuffle=True)
        testloader = DataLoader(testset,
                                shuffle=False, num_workers=1)

        # Display image
        def imshow(img):
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        # dataiter = iter(trainloader)
        # images, labels = dataiter.next()
        # imshow(torchvision.utils.make_grid(images))
        print("DATA LOADING SUCCESSFUL")
        # wait = input("PRESS ENTER TO START TRAINING")

        # --------------------------------------
        # CNN:
        use_gpu = torch.cuda.is_available()
        print("---------------- GPU IS : ", use_gpu)
        # print(n_conv, dim1, kernel_conv, stride_conv, kernel_pool, stride_pool, n_layers, dim2)
        # print('-------------------------')
        net = Net(n_conv, dim1, kernel_conv, stride_conv, kernel_pool, stride_pool, n_layers, dim2)
        print(net)

        if use_gpu:
            net = net.cuda()

        # Loss function
        # with optim, can also use e.g. Adam
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

        # --------------------------------------
        # Training:
        losslst = []
        starttime = time.time()
        traintime = time.time() - starttime
        epoch = 0
        # for epoch in range(nr_epochs):  # loop over the dataset multiple times
        while traintime < maxtraintime:
            epoch = epoch + 1
            print('epoch %d:' % (epoch))
            #            pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            #            print(pytorch_total_params)
            running_loss_epoch = 0.0  # reset running loss per epoch
            running_loss = 0.0  # reset running loss per batch
            # self.loss = min(losslst)/len(trainloader)

            #            pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            #            print(pytorch_total_params)
            running_loss_epoch = 0.0  # reset running loss per epoch
            running_loss = 0.0  # reset running loss per batch
            # self.loss = min(losslst)/len(trainloader)
            if epoch > minepoch:  # minimum number of epochs
                rule = abs(np.mean(np.diff(losslst[-5:]))) / losslst[-5:][0]
                if rule < convergence:
                    self.losslst = losslst
                    self.realtime = time.time() - starttime
                    break
            print('epoch %d:' % (epoch))
            for i, data in enumerate(trainloader, 0):  # for every batch, start at 0
                # get the inputs and labels
                if traintime > maxtraintime:
                    break
                inputs, labels = data
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)  # labels need to be of type: torch.long
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:  # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch, i + 1, running_loss / 200))
                    running_loss_epoch += running_loss
                    running_loss = 0.0
                #                    print(traintime)
                if epoch == 1 and i == 0:
                    batchtime = time.time() - starttime
                traintime = time.time() - starttime + batchtime  # + batchtime estimates the time for the next batch
                self.realtime = time.time() - starttime
            losslst.append(running_loss_epoch)
            self.tot_epoch = epoch
            self.losslst = losslst
            print(abs(np.mean(np.diff(losslst[-5:]))) / losslst[-5:][0])

        # --------------------------------
        # Testing:
        # Whole test data set
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.accuracy = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            self.accuracy))

        print('Finished Training')


'''
# ---------------------------------------------------------------------

# Main:
# Those 3 variables are set in the class itself look # parameters
# batch_size = 10
# lr = 0.001
# momentum = 0.9


# n_conv = 3
# dim1 = [6, 16, 32]
# kernel_conv = [3, 3, 3]
# stride_conv = [1, 1, 1]
# kernel_pool = [2, 2, 2]
# stride_pool = [2, 2, 2]
# n_layers = 3
# dim2 = [120, 84, 40]
# print('LOADING DATA...')
#
# image_path = 'database/'
# normalise = True  # Will transform [0, 255] to [0, 1]
# # Load data set and organise into batch size and right input for Net()
# dataset = CustomDataset(image_path=image_path, normalise=normalise, train=True)
# lengths = [10000, 10778]  # train data and test data
# train_dataset, test_dataset = random_split(dataset, lengths)  # 20778
# trial = CNN(train_dataset,
#             test_dataset,
#             n_conv,
#             dim1,
#             kernel_conv,
#             stride_conv,
#             kernel_pool,
#             stride_pool,
#             n_layers,
#             dim2)
=======
#batch_size = 10
#lr = 0.001
#momentum = 0.9

n_conv = 3
dim1 = [6, 19, 32]
kernel_conv = [3, 3, 3]
stride_conv = [2, 2, 2]
kernel_pool = [3, 3, 3]
stride_pool = [2, 2, 2]
n_layers = 10
dim2 = [120, 111, 102, 93, 84, 75, 66, 57, 48, 40]


print('LOADING DATA...')



image_path = 'database/'
normalise = True # Will transform [0, 255] to [0, 1]
# Load data set and organise into batch size and right input for Net()
dataset = CustomDataset(image_path=image_path, normalise=normalise, train=True)
lengths = [10000,10778] #train data and test data
train_dataset, test_dataset = random_split(dataset,lengths) # 20778
trial = CNN(train_dataset, test_dataset,
    n_conv,
    dim1,
    kernel_conv,
    stride_conv,
    kernel_pool,
    stride_pool,
    n_layers,
    dim2)
'''
# important self.variables are those three:
# print((trial.losslst)) # list of loss after each epoch
# print(trial.realtime)
# print(trial.accuracy)

# important self.variables are those three:
# print((trial.losslst)) # list of loss after each epoch
# print(trial.realtime)
# print(trial.accuracy)
