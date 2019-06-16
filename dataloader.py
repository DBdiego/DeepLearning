import os
import re
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataAugmentation import add_gaussian_noise



class CustomDataset(Dataset):
    """ Custom car logo Dataset loader"""

    # Labels:
    cars = ['Alfa Romeo', 'Audi', 'BMW', 'Chevrolet', 'Citroen', 'Dacia', 'Daewoo', 'Dodge',
            'Ferrari', 'Fiat', 'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Kia', 'Lada',
            'Lancia', 'Land Rover', 'Lexus', 'Maserati', 'Mazda', 'Mercedes', 'Mitsubishi',
            'Nissan', 'Opel', 'Peugeot', 'Porsche', 'Renault', 'Rover', 'Saab', 'Seat',
            'Skoda', 'Subaru', 'Suzuki', 'Tata', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo']
            
    # Initialise: load images and get labels
    def __init__(self, image_path, normalise, maxx, tot_imgs, resize=(224, 224), train=True, ):
        imgs = os.listdir(image_path)
        imgs = imgs[0:tot_imgs]
        n_samples = int(maxx/2)
        imgs_red = np.random.choice(imgs, n_samples, replace=False)
        self.train = train

        # Transforms: resize t0 224, torch tensor, normalise
        self.transform = transforms.Compose([ transforms.Resize(resize),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0, 0, 0), (255, 255, 255))])

        # Load images from image to np array -------------------
        images_og = np.array(
            [np.array(Image.open(image_path + img).convert("RGB")) for img in imgs_red],
            order='F', dtype='float32')  # uint8
        images_aug = add_gaussian_noise(images_og)
        self.images = np.concatenate((images_og, images_aug))

        # extracting labels from image names -------------------
        labels = np.array([self.cars.index(re.match(r"(^\D+)", img)[0]) for img in imgs_red])
        labels = torch.from_numpy(np.concatenate((labels, labels)))
        self.labels = labels.type(torch.long)  # outputs of network are also torch.long type...
        self.length = np.shape(self.images)[0]

    def __getitem__(self, item):
        img = self.images[item]
        img = Image.fromarray((img * 255).astype(np.uint8)) # Need to be ints
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
