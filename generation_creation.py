import numpy as np
from CNN_Class import CNN, random_split, CustomDataset
import torch
import time

import torch.multiprocessing as mp
from torch.multiprocessing import Process

def f(cnn_class_inputs, network_index,results):
    a = CNN(network_index,
            cnn_class_inputs[0],
            cnn_class_inputs[1],
            cnn_class_inputs[2],
            cnn_class_inputs[3],
            cnn_class_inputs[4],
            cnn_class_inputs[5],
            cnn_class_inputs[6],
            cnn_class_inputs[7],
            cnn_class_inputs[8],
            cnn_class_inputs[9],
            cnn_class_inputs[10])
    results[network_index] = a.accuracy








if __name__ == '__main__':
    #--------------------------------------------------------
    NORMALIZE = True
    IMAGE_PATH = 'database/'
    RATIO_TRAINING = 0.1
    RATIO_DATA = 0.1
    MAX_DATA = RATIO_DATA * 41556

    print('Running gen_creat.py\n')

    print('Importing data: ...')
    dataset = CustomDataset(image_path=IMAGE_PATH, normalise=NORMALIZE, maxx=MAX_DATA, train=True)
    print('Importing data: DONE\n')

    I = int(RATIO_TRAINING * len(dataset))
    lengths = [len(dataset) - I, I]  # train data and test data
    train_dataset, test_dataset = random_split(dataset, lengths)

    if torch.cuda.is_available():
        num_avail_gpus = torch.cuda.device_count()
    else:
        num_avail_gpus = 1

    genome1 = [5, [82, 161, 241, 320, 400], [2, 2, 2, 2, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], 7,
               [200, 173, 146, 120, 93, 66, 40]]
    genome2 = [6, [69, 135, 201, 267, 333, 400], [2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
               [2, 2, 2, 2, 2, 2], 7, [200, 173, 146, 120, 93, 66, 40]]
    genomes = [genome1, genome2, genome2, genome1]

    args = []
    for i in range(len(genomes)):

        gpu_index = i%num_avail_gpus#np.random.randint(0,2)
        #print(gpu_index)
        args.append(
            [gpu_index, train_dataset, test_dataset, genomes[i][0], genomes[i][1], genomes[i][2], genomes[i][3], genomes[i][4],
             genomes[i][5], genomes[i][6], genomes[i][7]])

    # --------------------------------------------------------
    #mp.get_context('spawn')
    mp.set_start_method('spawn',force=True)
    manager = mp.Manager()
    results = manager.dict()

    def create_pocess(gen_index,processes,network_index):
        p = mp.Process(target=f, args=(args[gen_index],network_index,results))
        p.start()
        processes.append([gen_index,p])

        return processes

    for j in range(int(len(genomes)/num_avail_gpus)):
        processes = []
        print('Beginning new cycle...')
        for i in range(num_avail_gpus):
            processes = create_pocess(i,processes,int(str(j)+str(i),2))

        for i in range(num_avail_gpus):
            p = processes[i][1]
            p.join()
            print('\tNetwork',int(str(j)+str(i),2),'done')

        print('Cycle done\n')

    print(results)
            #processes.pop(i)






