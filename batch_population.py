from external.sga.population import *
from paralleliser import fitness_func 


class BatchPopulation(Population):

    def __init__(self, m: int, n: int, chromosome_class: Chromosome):
        super().__init__(m, n, chromosome_class)

    def crossover(self, crossover_function, *args):
        children_population = BatchPopulation(self.m, self.n, self._chromosome)
        children_population.contestants = crossover_function(self._population, self.n, *args)
        return children_population

    def calculate_fitness(self, fitness_function, *args, **kwargs):
        result = [None for i in self._population]
        genomes = []
        for index, member in enumerate(self._population):
            
            # If already computed previously
            if kwargs["archive"]["chromosome"].str.contains(member).any():
                result[index] = kwargs["archive"]["fitness"][kwargs["archive"]["chromosome"] == member].iloc[0]


            else:
                genomes.append(self._chromosome.parameters(member))

        
        result = fitness_func(genomes, kwargs['train_dataset'], kwargs['test_dataset'], result)
        print(result)
        return result
