import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from CNN_Class import CNN, random_split, CustomDataset

from cnn2 import Net
from dataloader import CustomDataset
from LogCreator import Add_to_Log, get_run_id







if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    network_index = 1
    generation_index = 1
    gpu_index = 0

    NORMALIZE = True
    IMAGE_PATH = 'database/'
    no_classes = [5, 8, 10, 20, 40]
    imgs_classes = [3299, 4296, 5434, 10521, 20778]  # number of images for number of classes above
    CLASSES_INDEX = 0  # NOTE: have to change line 18 in batch_population as well
    RATIO_TESTING = 0.3
    RATIO_DATA = 1
    MAX_DATA = RATIO_DATA * 2 * imgs_classes[CLASSES_INDEX]  # 41556


    # Creating ID for this simulation run
    run_ID = get_run_id(status='create_new')
    print('RUN ID: ', run_ID, '\n')

    print('Importing data: ...')
    dataset = CustomDataset(image_path=IMAGE_PATH, normalise=NORMALIZE, maxx=MAX_DATA,
                            tot_imgs=imgs_classes[CLASSES_INDEX], resize=(50, 50))
    print('Importing data: DONE\n')

    I = int(RATIO_TESTING * len(dataset))
    lengths = [len(dataset) - I, I]  # train data and test data
    train_dataset, test_dataset = random_split(dataset, lengths)



    n_conv = 3
    dim1 = [16,32,64]
    kernel_conv = [3,3,3]
    stride_conv = [1,1,1]
    kernel_pool = [2,2,2]
    stride_pool = [2,2,2]
    n_layers = 3
    dim2 = [500,500,5]


    CNN(network_index, generation_index, gpu_index, train_dataset, test_dataset,
            n_conv,
            dim1,
            kernel_conv,
            stride_conv,
            kernel_pool,
            stride_pool,
            n_layers,
            dim2)



