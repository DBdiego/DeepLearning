import os
import re
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np



class CustomDataset(Dataset):
    """ Custom car logo Dataset loader"""

    # Labels:
    cars = ['Alfa Romeo', 'Audi', 'BMW', 'Chevrolet', 'Citroen', 'Dacia', 'Daewoo', 'Dodge',
            'Ferrari', 'Fiat', 'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Kia', 'Lada',
            'Lancia', 'Land Rover', 'Lexus', 'Maserati', 'Mazda', 'Mercedes', 'Mitsubishi',
            'Nissan', 'Opel', 'Peugeot', 'Porsche', 'Renault', 'Rover', 'Saab', 'Seat',
            'Skoda', 'Subaru', 'Suzuki', 'Tata', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo']

    # Initialise: load images and get labels
    def __init__(self, image_path, normalise, resize=(50, 50), train=True):
        imgs = os.listdir(image_path)
        n_samples = np.size(imgs)
        self.train = train

            
        # if normalising (to [0, 1]...), needs to be float type
        if normalise:
            # define transforms from numpy array to pytorch tensor, and optionally normalising
            self.transform = transforms.Compose([ transforms.Resize(resize),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0, 0, 0), (255, 255, 255))])
            # transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))]) # or to [-1, 1]

            # Load images from image to np array.
            self.images = np.array(
                [np.array(Image.open(image_path + img).convert("RGB")) for img in os.listdir(image_path)],
                order='F', dtype='float32')  # uint8
        else:  # without normalisation...
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.images = np.array(
                [np.array(Image.open(image_path + img).convert("RGB")) for img in os.listdir(image_path)],
                order='F', dtype='uint8')  # float32
        # extracting labels from image names
        labels = torch.from_numpy(
            np.array([self.cars.index(re.match(r"(^\D+)", imgs[i])[0]) for i in range(n_samples)]))
        self.labels = labels.type(torch.long)  # outputs of network are also torch.long type...
        self.length = np.shape(self.images)[0]

    def __getitem__(self, item):
        img = self.images[item]
        # applying transformations...
        # if self.transform is not None:
        img = Image.fromarray((img * 255).astype(np.uint8))
        img = self.transform(img)
        lbl = self.labels[item]
        return img, lbl

    def __len__(self):
        return self.length


'''
---------- INCLUDE--------------- :
from dataloader import CustomDataset
from torch.utils.data import DataLoader

---------- LOADING DATA---------- :
trainset = CustomDataset(image_path=image_path, normalise=True, train=True)
trainloader = DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True)

---------- GET SAMPLE BATCH------ :
dataiter = iter(trainloader)
images, labels = dataiter.next()

---------- ITERATE THROUGH------- :
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
'''
