# from external.sga.population import *
from batch_population import *
from external.sga.strategy import *
from external.sga.operators import *
from external.sga.gene import *
from external.sga.chromosome import *
import matplotlib.pyplot as plt
from CNN_Class import random_split, CustomDataset
import torch


NORMALIZE = True
IMAGE_PATH = 'database/'
no_classes = [5,8,10,20,40]
imgs_classes = [3299,4296,5434,10521,20778] # number of images for number of classes above
CLASSES_INDEX = 0 # NOTE: have to change line 18 in batch_population as well
RATIO_TRAINING = 0.9
RATIO_DATA = 1
MAX_DATA = RATIO_DATA * 2 * imgs_classes[CLASSES_INDEX]#41556

POP_SIZE = 10
NUM_GENERATIONS = 20


def load_data():

    print('Importing data: ...')
    dataset = CustomDataset(image_path=IMAGE_PATH, normalise=NORMALIZE, maxx=MAX_DATA, tot_imgs=imgs_classes[CLASSES_INDEX])
    print('Importing data: DONE\n')

    I = int(RATIO_TRAINING * len(dataset))
    lengths = [len(dataset) - I, I]  # train data and test data
    train_dataset, test_dataset = random_split(dataset, lengths)
    return train_dataset, test_dataset


# Define the Chromosome which maps to a solution.
ChromosomePart2 = Chromosome(
    [
        # LinearRangeGene(-1, 1, 100),  # k2
        DenaryGeneFloat(limits=(4, 6), n_bits_exponent=3, n_bits_fraction=None, signed=False),  # num. conv layers
        DenaryGeneFloat(limits=(2, 3), n_bits_exponent=2, n_bits_fraction=None, signed=False),  # kernel size conv layers
        DenaryGeneFloat(limits=(1, 1), n_bits_exponent=2, n_bits_fraction=None, signed=False),  # stride conv layers
        DenaryGeneFloat(limits=(2, 3), n_bits_exponent=2, n_bits_fraction=None, signed=False),  # kernel size pool layers
        DenaryGeneFloat(limits=(2, 2), n_bits_exponent=2, n_bits_fraction=None, signed=False),  # stride pool layers
        DenaryGeneFloat(limits=(3, 5), n_bits_exponent=3, n_bits_fraction=None, signed=False),  # num .neurons FCNN layers
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
    torch.multiprocessing.freeze_support()
    train_dataset, test_dataset = load_data()

    # Evolve for solution.
    EvolutionaryStrategyTest.evolve(True, 
        train_dataset = train_dataset,
        test_dataset = test_dataset
    )

    sol = EvolutionaryStrategyTest.get_fittest_solution()[0]
    print(sol)

