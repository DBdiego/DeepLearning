# from external.sga.population import *
from batch_population import *
from external.sga.strategy import *
from external.sga.operators import *
from external.sga.gene import *
from external.sga.chromosome import *
import matplotlib.pyplot as plt
from CNN_Class import random_split, CustomDataset
from LogCreator import get_run_id
import torch
import os


NORMALIZE     = True
IMAGE_PATH    = 'database/'
no_classes    = [5,8,10,20,40]
imgs_classes  = [3299,4296,5434,10521,20778] # number of images for number of classes above
CLASSES_INDEX = 0 # NOTE: have to change line 18 in batch_population as well
RATIO_TESTING = 0.3
RATIO_DATA    = 1
MAX_DATA      = RATIO_DATA * 2 * imgs_classes[CLASSES_INDEX]#41556

POP_SIZE = 12
NUM_GENERATIONS = 20


def load_data():
    print('Importing data: ...')
    dataset = CustomDataset(image_path=IMAGE_PATH, normalise=NORMALIZE, maxx=MAX_DATA, tot_imgs=imgs_classes[CLASSES_INDEX])
    print('Importing data: DONE\n')

    I = int(RATIO_TESTING * len(dataset))
    lengths = [len(dataset)-I, I]  # train data and test data
    train_dataset, test_dataset = random_split(dataset, lengths)
    return train_dataset, test_dataset


# Define the Chromosome which maps to a solution.
ChromosomePart2 = Chromosome(
    [
        # LinearRangeGene(-1, 1, 100),  # k2
        DenaryGeneFloat(limits=(4, 7), n_bits_exponent=3, n_bits_fraction=None, signed=False),  # num. conv layers
        DenaryGeneFloat(limits=(2, 7), n_bits_exponent=3, n_bits_fraction=None, signed=False),  # kernel size conv layers
        #DenaryGeneFloat(limits=(1, 1), n_bits_exponent=1, n_bits_fraction=None, signed=False),  # stride conv layers
        DenaryGeneFloat(limits=(2, 4), n_bits_exponent=3, n_bits_fraction=None, signed=False),  # kernel size pool layers
        DenaryGeneFloat(limits=(2, 3), n_bits_exponent=2, n_bits_fraction=None, signed=False),  # stride pool layers
        DenaryGeneFloat(limits=(3, 7), n_bits_exponent=3, n_bits_fraction=None, signed=False),  # num .neurons FCNN layers
    ],
)

# Create population.
population = BatchPopulation(POP_SIZE, POP_SIZE, ChromosomePart2)

# Define termination class
TerminationCriteria = TerminationCriteria()
# TerminationCriteria.add_convergence_limit()  # Limit to 0.1% convergence in population.
TerminationCriteria.add_generation_limit(NUM_GENERATIONS)


def fitness_function(*args):
    return 0


# Evolutionary Strategy Tests
EvolutionaryStrategyTest = EvolutionaryStrategy(population=population,
                                                fitness_function=fitness_function,
                                                crossover_function=CrossoverOperator.random_polygamous,
                                                selection_function=SelectionOperator.supremacy,
                                                termination_criteria=TerminationCriteria.check,
                                                mutation_rate=0.1,
                                                )






if __name__ == '__main__':
    # Creating ID for this simulation run
    run_ID = get_run_id(status='create_new')
    print('RUN ID: ', run_ID, '\n')

    # Adding freeze support for pytorch.multiprocessing module
    torch.multiprocessing.freeze_support()

    # Loading Dataset
    train_dataset, test_dataset = load_data()

    # Evolve for solution.
    ENN_log = EvolutionaryStrategyTest.evolve(verbose    = True ,
                                              return_log = True ,
                                              train_dataset = train_dataset,
                                              test_dataset  = test_dataset ,
                                              run_id = run_ID)

    # Determining Final Solution
    solution = EvolutionaryStrategyTest.get_fittest_solution()[0]
    print(solution)
    

    #Saving Logs to backup folder with run ID as filename
    f = open('./Logs/Generation_Logs/'+str(run_ID)+'_Generations_Logs.csv', 'r')
    run_logs = f.readlines()
    f.close()
    
    f = open('./Logs/Backup_Logs/'+str(run_ID)+'.csv', 'w')
    f.write('\n'.join(run_logs))
    f.close()

    os.remove('./Logs/Generation_Logs/'+str(run_ID)+'_Generations_Logs.csv') 







    
    

