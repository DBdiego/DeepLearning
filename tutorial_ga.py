# from external.sga.population import *
from batch_population import *
from external.sga.strategy import *
from external.sga.operators import *
from external.sga.gene import *
from external.sga.chromosome import *
import matplotlib.pyplot as plt
from generation_creation import paralalala
from CNN_Class import random_split, CustomDataset
import torch

def load_data():
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
    return train_dataset, test_dataset


# Define the Chromosome which maps to a solution.
ChromosomePart2 = Chromosome(
    [
        # LinearRangeGene(-1, 1, 100),  # k2
        DenaryGeneFloat(limits=(2, 8), n_bits_exponent=3, n_bits_fraction=None, signed=False),  # k2
        DenaryGeneFloat(limits=(2, 3), n_bits_exponent=2, n_bits_fraction=None, signed=False),  # k2
        DenaryGeneFloat(limits=(1, 2), n_bits_exponent=2, n_bits_fraction=None, signed=False),  # k2
        DenaryGeneFloat(limits=(2, 3), n_bits_exponent=2, n_bits_fraction=None, signed=False),  # k2
        DenaryGeneFloat(limits=(2, 3), n_bits_exponent=2, n_bits_fraction=None, signed=False),  # k2
        DenaryGeneFloat(limits=(4, 16), n_bits_exponent=5, n_bits_fraction=None, signed=False),  # k2
    ],
)

# Create population.
population = BatchPopulation(100, 100, ChromosomePart2)

# Define termination class
TerminationCriteria = TerminationCriteria()
# TerminationCriteria.add_convergence_limit()  # Limit to 0.1% convergence in population.
TerminationCriteria.add_generation_limit(20)


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
    EvolutionaryStrategyTest.evolve(True, {
        "training_dataset": train_dataset,
        "test_dataset": test_dataset
    })

    sol = EvolutionaryStrategyTest.get_fittest_solution()[0]
    print(sol)

