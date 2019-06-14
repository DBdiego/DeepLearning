import numpy as np
from CNN_Class import CNN, random_split, CustomDataset
import torch
import time

import torch.multiprocessing as mp

# Runs the CNN and passes the accuracy to results in paralalala
def f(cnn_class_inputs, network_index, results):
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

'''
Trains each network on a different GPU
note: number of genomes passed has to be a multiple of number of available GPUs
else will raise ValueError
Inputs:
    genomes:         list of genomes
    train_dataset:   CustomDataset training data
    test_dataset:    CustomDataset training data
Outputs:
    results:         list of accuracies of the CNN with the genomes given
'''
def paralalala(genomes,train_dataset, test_dataset):

    if __name__ == 'generation_creation':#''__main__':


        # Set up variables and proper input layout -------------------------------------------------------
        if torch.cuda.is_available():
            num_avail_gpus = torch.cuda.device_count()
        else:
            num_avail_gpus = 1

        num_cycles = int(len(genomes) / num_avail_gpus)
        if len(genomes) % num_avail_gpus != 0:
            num_cycles += 1

        args = []
        for i in range(len(genomes)):

            gpu_index = i%num_avail_gpus
            args.append(
                [gpu_index, train_dataset, test_dataset, genomes[i][0], genomes[i][1], genomes[i][2], genomes[i][3], genomes[i][4],
                 genomes[i][5], genomes[i][6], genomes[i][7]])

        # --------------------------------------------------------
        mp.set_start_method('spawn',force=True)
        manager = mp.Manager()
        results = manager.dict()

        # Creates, starts process and appends it to list processes
        def create_pocess(processes,network_index):
            p = mp.Process(target=f, args=(args[network_index],network_index,results))
            p.start()
            processes.append([network_index,p])

            return processes

        for j in range(num_cycles):
            processes = []
            print('Beginning new cycle...')
            if j!=num_cycles-1:
                num_used_gpus = num_avail_gpus
            else:
                num_used_gpus = len(genomes)%num_avail_gpus
            for i in range(num_used_gpus):
                processes = create_pocess(processes,int(str(j)+str(i),2))

            for i in range(num_used_gpus):
                p = processes[i][1]
                p.join()
                print('\tNetwork',int(str(j)+str(i),2),'done')

            print('Cycle done\n')

    return results




