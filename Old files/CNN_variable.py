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
    #----------
    #  Loading data:
    transform = transforms.Compose(
        [transforms.ToTensor()])#,
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=1)

    classes = ('0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9')



    #----------
    # Functions to show an image
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images


    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))



    #----------
    # CNN:

    use_gpu = torch.cuda.is_available()

    # Defining custom nn module

    dim = [1,6,16]
    final_dim = 7
    dim2 = [16 * final_dim * final_dim,120,84,10]

    class Net(nn.Module):
        def __init__(self, n_c_layers, n_l_layers):
            self.dim=np.linspace(1,16, n_c_layers+1)
            super(Net, self).__init__()

            # Defining convolution layers
            self.n_c_layers = n_c_layers
            for i in range(n_c_layers):
                setattr(self, f"conv{i}", nn.Conv2d(int(dim[i]), int(dim[i+1]), 3, padding=1))
                self.pool = nn.MaxPool2d(2, 2) # (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

            # Defining network layers
            self.n_l_layers = n_l_layers
            for i in range(n_l_layers):
                setattr(self, f"fc{i}", nn.Linear(int(dim2[i]), int(dim2[i+1])))


        def forward(self, x):
            # Convolution layers:
            for i in range(self.n_c_layers):
                x=self.pool(F.relu(getattr(self, f"conv{i}")(x)))

            print(final_dim)
            x = x.view(-1, 16 * final_dim * final_dim) # image to tensor structure ?

            # Neural net layers:
            for i in range(self.n_l_layers):
                if i != self.n_l_layers-1:
                    x = F.relu(getattr(self, f"fc{i}")(x))
                else:
                    x = getattr(self, f"fc{i}")(x)
            return x


    net = Net(2,3)
    print(net)
    if use_gpu:
        net = net.cuda()

    # Loss function
    # with optim, can also use e.g. Adam
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0): # for every image, start at 0
            # get the inputs
            inputs, labels = data
            #if i == 2000:
            #    imshow(torchvision.utils.make_grid(inputs))
            #    print(inputs.shape)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')



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


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
