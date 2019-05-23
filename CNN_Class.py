from dataloader import CustomDataset
from cnn2 import Net
from torch.utils.data import DataLoader

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

#torch.manual_seed(0)




class CNN:

    def __init__(self, trainset,
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
        nr_epochs = 100
        batch_size = 10
        lr = 0.001
        momentum = 0.9

        # --------------------------------------
        # Data loading:
        trainloader = DataLoader(dataset=trainset,
                                 batch_size=batch_size,
                                 shuffle=True)
        # Display image
        def imshow(img):
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        #dataiter = iter(trainloader)
        #images, labels = dataiter.next()
        #imshow(torchvision.utils.make_grid(images))
        print("DATA LOADING SUCCESSFUL")
        wait = input("PRESS ENTER TO START TRAINING")


        #--------------------------------------
        # CNN:
        use_gpu = torch.cuda.is_available()
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
        running_loss_epoch = 1000000
        losslst = []
        for epoch in range(nr_epochs):  # loop over the dataset multiple times
            print('epoch {}:'.format(epoch + 1))
            losslst.append(running_loss_epoch)
            running_loss_epoch = 0.0
            running_loss = 0.0
            self.loss = min(losslst)/len(trainloader)
            if epoch > 5:  # minimum number of epochs
                if (losslst[epoch] - losslst[epoch - 1]) / losslst[epoch - 1] * 100 < 0.001:
                    break
            for i, data in enumerate(trainloader, 0):  # for every batch, start at 0
                # get the inputs and labels
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
                          (epoch + 1, i + 1, running_loss / 200))
                    running_loss_epoch += running_loss
                    running_loss = 0.0

            self.tot_epoch = epoch


        print('Finished Training')




# ---------------------------------------------------------------------

# Main:
nr_epochs = 5
batch_size = 10
lr = 0.001
momentum = 0.9
n_conv = 2
dim1 = [6, 16]
kernel_conv = [3, 3]
stride_conv = [1, 1]
kernel_pool = [2, 2]
stride_pool = [2, 2]
n_layers = 3
dim2 = [120, 84, 40]
print('LOADING DATA...')
image_path = 'images/'
normalise = True # Will transform [0, 255] to [0, 1]
# Load data set and organise into batch size and right input for Net()
trainset = CustomDataset(image_path=image_path, normalise=normalise, train=True)
trial = CNN(trainset,
    n_conv,
    dim1,
    kernel_conv,
    stride_conv,
    kernel_pool,
    stride_pool,
    n_layers,
    dim2)
print(trial.tot_epoch)
print(trial.loss)