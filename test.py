from generation_creation import paralalala
from CNN_Class import CNN, random_split, CustomDataset
import torch

def main():
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
    genome1 = [5, [82, 161, 241, 320, 400], [2, 2, 2, 2, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2], 7,
               [200, 173, 146, 120, 93, 66, 40]]
    genome2 = [6, [69, 135, 201, 267, 333, 400], [2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
               [2, 2, 2, 2, 2, 2], 7, [200, 173, 146, 120, 93, 66, 40]]
    genomes = [genome1, genome2, genome2, genome1]
    tit = paralalala(genomes,train_dataset, test_dataset)
    print(tit)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()

