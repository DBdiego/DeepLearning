import numpy as np
from CNN_Class import CNN, random_split, CustomDataset
import torch
import time
import datetime
from LogCreator import Add_to_Log, get_run_id
import os

import torch.multiprocessing as mp



# Runs the CNN and passes the accuracy to results in fitness_func()
def train_CNN(cnn_class_inputs, network_index, generation_index, results):
    print(cnn_class_inputs[3:])
    network = CNN(  network_index        ,
                    generation_index     ,
                    cnn_class_inputs[ 0] ,
                    cnn_class_inputs[ 1] ,
                    cnn_class_inputs[ 2] ,
                    cnn_class_inputs[ 3] ,
                    cnn_class_inputs[ 4] ,
                    cnn_class_inputs[ 5] ,
                    cnn_class_inputs[ 6] ,
                    cnn_class_inputs[ 7] ,
                    cnn_class_inputs[ 8] ,
                    cnn_class_inputs[ 9] ,
                    cnn_class_inputs[10] )

    results[network_index] = network.all_info

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
def fitness_func(genomes, generation_index,train_dataset, test_dataset, results_HL):


    # Set up variables and proper input layout -------------------------------------------------------
    if torch.cuda.is_available():
        num_avail_gpus = torch.cuda.device_count()
        if num_avail_gpus > 2:
            num_avail_gpus = 3
    else:
        num_avail_gpus = 1

    num_cycles = int(len(genomes) / num_avail_gpus)
    if len(genomes) % num_avail_gpus != 0:
        num_cycles += 1

    args = []
    for i in range(len(genomes)):

        gpu_index = i%num_avail_gpus
        args.append([gpu_index,
                     train_dataset,
                     test_dataset,
                     genomes[i][0],
                     genomes[i][1],
                     genomes[i][2],
                     genomes[i][3],
                     genomes[i][4],
                     genomes[i][5],
                     genomes[i][6],
                     genomes[i][7]])

    # --------------------------------------------------------
    mp.set_start_method('spawn',force=True)
    manager = mp.Manager()
    results = manager.dict()

    # Creates, starts process and appends it to list processes
    def create_pocess(processes,network_index, generation_index):
        p = mp.Process(target=train_CNN, args=(args[network_index],network_index,generation_index, results))
        p.start()
        processes.append([network_index,p, (datetime.datetime.now()+datetime.timedelta(hours=2))])
        return processes

    log_dict = {}
    counter = 0
    for j in range(num_cycles):
        processes = []
        print('Loading GPUs...')
        if j!=num_cycles-1:
            num_used_gpus = num_avail_gpus
        else:
            num_used_gpus = len(genomes)-j*num_avail_gpus

        for i in range(num_used_gpus):
            processes = create_pocess(processes, counter, generation_index)
            counter+=1

        for i in range(num_used_gpus):
            p = processes[i][1]
            network_id = processes[i][0]
            p.join()
        
            print('\tNetwork',network_id,'DONE')

        print('Cycle done\n')


    # Saving results
    log_dict = {}
    run_ID = get_run_id()
    j = 0
    results_final = results.copy()
    for i in range(len(results_HL)):
        if results_HL[i] == None:
            results_HL[i] = results_final[j]['accuracy']
            Add_to_Log(results_final[j], './Logs/Generation_Logs/'+str(run_ID)+'_Generations_Logs.csv')
            j += 1

    # Clearing GPU logs
    for file_name in os.listdir('./Logs/GPU_Logs'):
        f = open('./Logs/GPU_Logs/'+file_name, 'w')
        f.write('')
        f.close()
    

    return results_HL




