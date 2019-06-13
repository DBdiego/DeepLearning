from ga.population import *
from ga.strategy import *
from ga.operators import *
from ga.gene import *
from ga.chromosome import *
import matplotlib.pyplot as plt

VAR1_LIMS = (3, 10)

# Define the Chromosome which maps to a solution.
ChromosomePart2 = Chromosome(
    [
        # LinearRangeGene(-1, 1, 100),  # k2
        DenaryGeneFloat(limits=VAR1_LIMS, n_bits_exponent=4, n_bits_fraction=None, signed=False),  # k2
        # DenaryGeneFloat(limits=VAR1_LIMS, n_bits_exponent=4, n_bits_fraction=0, signed=False),  # k2
        # DenaryGeneFloat(limits=VAR1_LIMS, n_bits_exponent=4, n_bits_fraction=0, signed=False),  # k2
        # DenaryGeneFloat(limits=VAR1_LIMS, n_bits_exponent=4, n_bits_fraction=0, signed=False),  # k2
        # DenaryGeneFloat(limits=VAR1_LIMS, n_bits_exponent=4, n_bits_fraction=0, signed=False),  # k2
        # DenaryGeneFloat(limits=VAR1_LIMS, n_bits_exponent=4, n_bits_fraction=0, signed=False),  # k2
    ],
)

# Create population.
population = BatchPopulation(100, 100, ChromosomePart2)

# Define termination class
TerminationCriteria = TerminationCriteria()
# TerminationCriteria.add_convergence_limit()  # Limit to 0.1% convergence in population.
TerminationCriteria.add_generation_limit(20)


def fitness_function(x):
    return 1 + x - x ** 2 + x ** 3 - x ** 4


# Evolutionary Strategy Tests
EvolutionaryStrategyTest = EvolutionaryStrategy(population=population,
                                                fitness_function=fitness_function,
                                                crossover_function=CrossoverOperator.random_polygamous,
                                                selection_function=SelectionOperator.supremacy,
                                                termination_criteria=TerminationCriteria.check,
                                                mutation_rate=0.1,
                                                )

# Evolve for solution.
EvolutionaryStrategyTest.evolve(verbose=True)

sol = EvolutionaryStrategyTest.get_fittest_solution()[0]
print(sol)
