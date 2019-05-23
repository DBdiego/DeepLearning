from dataloader import CustomDataset
from cnn2 import Net
from torch.utils.data import DataLoader, random_split

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#torch.manual_seed(0)




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
        nr_epochs = 20
        batch_size = 10
        lr = 0.001
        momentum = 0.9

        # --------------------------------------
        # Data loading:
        trainloader = DataLoader(dataset=trainset,
                                 batch_size=batch_size,
                                 shuffle=True)
        testloader = DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=1)
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
        # running_loss_epoch = 1000000
        losslst = []
        for epoch in range(nr_epochs):  # loop over the dataset multiple times
            print('epoch {}:'.format(epoch + 1))
#            losslst.append(running_loss_epoch)
            running_loss_epoch = 0.0
            running_loss = 0.0
            #self.loss = min(losslst)/len(trainloader)
            if epoch > 6:  # minimum number of epochs
                rule = abs(np.mean(np.diff(losslst[-5:])))/losslst[-5:][0]
                
                #if abs(losslst[epoch] - losslst[epoch - 1]) / losslst[epoch - 1] * 100 < 0.001:
                if rule < 0.001:
#                    print(abs(losslst[epoch] - losslst[epoch - 1]) / losslst[epoch - 1] * 100)
                    self.losslst = losslst
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
            losslst.append(running_loss_epoch)
            self.tot_epoch = epoch
            self.losslst = losslst
            print(abs(np.mean(np.diff(losslst[-5:])))/losslst[-5:][0])

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

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


        print('Finished Training')




# ---------------------------------------------------------------------

# Main:
nr_epochs = 'poep'
batch_size = 10
lr = 0.001
momentum = 0.9
n_conv = 3
dim1 = [6, 16, 32]
kernel_conv = [3, 3, 3]
stride_conv = [1, 1, 1]
kernel_pool = [2, 2, 2]
stride_pool = [2, 2, 2]
n_layers = 3
dim2 = [120, 84, 40]
print('LOADING DATA...')



image_path = 'database/'
normalise = True # Will transform [0, 255] to [0, 1]
# Load data set and organise into batch size and right input for Net()
dataset = CustomDataset(image_path=image_path, normalise=normalise, train=True)
lengths = [10000,10778]
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
print(len(trial.losslst))
print((trial.losslst))
#print(trial.tot_epoch)
#print(trial.loss)

plt.plot(range(len(trial.losslst)),trial.losslst)
plt.show()
