import time
import os
import re

from PIL import Image # Image processing

from torch.utils.data import Dataset, DataLoader

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(0)


def main():

    # Parameters
    nr_epochs = 5
    batch_size = 10
    lr = 0.001
    momentum = 0.9

    # --------------------------------------------------
    # DATALOADER CLASS:

    image_path = 'images/'
    imgs = os.listdir(image_path)
    img_x = img_y = 50  # image size
    n_samples = np.size(imgs) # 20778

    class CustomDataset(Dataset):
        """ Custom car logo Dataset loader"""

        # Labels:
        cars = ['Alfa Romeo', 'Audi', 'BMW', 'Chevrolet', 'Citroen', 'Dacia', 'Daewoo', 'Dodge',
                'Ferrari', 'Fiat', 'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Kia', 'Lada',
                'Lancia', 'Land Rover', 'Lexus', 'Maserati', 'Mazda', 'Mercedes', 'Mitsubishi',
                'Nissan', 'Opel', 'Peugeot', 'Porsche', 'Renault', 'Rover', 'Saab', 'Seat',
                'Skoda', 'Subaru', 'Suzuki', 'Tata', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo']

        # Initialise: load images and get labels
        def __init__(self, image_path, normalise, train=True):
            self.train = train
            # if normalising (to [0, 1]...), needs to be float type
            if normalise:
                # define transforms from numpy array to pytorch tensor, and optionally normalising
                self.transform = transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0,0,0),(255,255,255))])
                                                #transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))]) # or to [-1, 1]

                # Load images from image to np array.
                self.images = np.array(
                    [np.array(Image.open(image_path + img).convert("RGB")) for img in os.listdir(image_path)],
                    order='F', dtype='float32') # uint8
            else: # without normalisation...
                self.transform = transforms.Compose([transforms.ToTensor()])
                self.images = np.array(
                    [np.array(Image.open(image_path + img).convert("RGB")) for img in os.listdir(image_path)],
                    order='F', dtype='uint8')  # float32
            # extracting labels from image names
            labels = torch.from_numpy(np.array([self.cars.index(re.match(r"(^\D+)", imgs[i])[0]) for i in range(n_samples)]))
            self.labels = labels.type(torch.long) # outputs of network are also torch.long type...
            self.length = np.shape(self.images)[0]

        def __getitem__(self, item):
            img = self.images[item]
            # applying transformations...
            #if self.transform is not None:
            img = self.transform(img)
            lbl = self.labels[item]
            return img, lbl

        def __len__(self):
            return self.length



    #----------
    # Will transform [0, 255] to [0, 1]
    normalise = True

    trainset = CustomDataset(image_path=image_path, normalise=normalise, train=True)
    trainloader = DataLoader(dataset = trainset,
                              batch_size = batch_size,
                              shuffle = True)

    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    #print(images.shape)
    imshow(torchvision.utils.make_grid(images))
    #print(labels)
    wait = input("PRESS ENTER TO CONTINUE.")



    #--------------------------------------
    # CNN:


    use_gpu = torch.cuda.is_available()

    # Parameters: [convolution, maxpooling]
    kernel_size = [3,2]
    padding = [1,0]
    stride = [1,2]
    dim1 = ((img_x - kernel_size[0] + 2 * padding[0]) / stride[0] + 1)
    dim2 = ((dim1 - kernel_size[1] + 2 * padding[1]) / stride[1] + 1)
    dim3 = ((dim2 - kernel_size[0] + 2 * padding[0]) / stride[0] + 1)
    dim4 = ((dim3 - kernel_size[1] + 2 * padding[1]) / stride[1] + 1)
    final_dim = int(dim4)
    # Defining custom nn module
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, kernel_size[0], padding=padding[0]) # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            self.pool = nn.MaxPool2d(2, 2) # (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            self.conv2 = nn.Conv2d(6, 16, kernel_size[0], padding=padding[0])
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * final_dim * final_dim, 120) # (in_features, out_features, bias=True), 7x7 is image size after conv & pooling...
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 40)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * final_dim * final_dim)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()
    print(net)
    if use_gpu:
        net = net.cuda()

    # Loss function
    # with optim, can also use e.g. Adam
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)


    for epoch in range(nr_epochs):  # loop over the dataset multiple times
        print('epoch {}:'.format(epoch+1))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0): # for every batch, start at 0
            # get the inputs and labels
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels) # labels need to be of type: torch.long
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')


    '''
    # Test on one image:
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

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
    '''

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()