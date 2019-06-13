import numpy as np
from CNN_Class import CNN, random_split, CustomDataset
import torch
import time
import torch.multiprocessing
from multiprocessing import Process


NORMALIZE = True
IMAGE_PATH = 'database/'
RATIO_TRAINING = 0.1
RATIO_DATA = 0.1
MAX_DATA = RATIO_DATA*41556

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
genome1 = [5, [82, 161, 241, 320, 400], [2, 2, 2, 2, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], 7, [200, 173, 146, 120, 93, 66, 40]]
genome2 = [6, [69, 135, 201, 267, 333, 400], [2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], 7, [200, 173, 146, 120, 93, 66, 40]]
genomes = [genome1,genome2]

args = []
for i in range(2):
    print(i)
    args.append([i, train_dataset, test_dataset, genomes[i][0], genomes[i][1], genomes[i][2], genomes[i][3], genomes[i][4],
        genomes[i][5], genomes[i][6], genomes[i][7]])


#torch.multiprocessing.spawn(CNN(), args=args[0], nprocs=2, join=True, daemon=False)



def f(cnn_class_inputs):
    CNN(cnn_class_inputs[0],
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

if __name__ == '__main__':
    p = Process(target=f, args=(args[0],))
    p1 = Process(target=f, args=(args[1],))
    p.start()
    p1.start()
    p.join()
    p1.join()





