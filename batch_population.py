from external.sga.population import *


class BatchPopulation(Population):

    def __init__(self, m: int, n: int, chromosome_class: Chromosome):
        super().__init__(m, n, chromosome_class)

    def crossover(self, crossover_function, *args):
        children_population = BatchPopulation(self.m, self.n, self._chromosome)
        children_population.contestants = crossover_function(self._population, self.n, *args)
        return children_population

    def calculate_fitness(self, fitness_function, *args, **kwargs):
        result = []
        for member in self._population:
            # print(kwargs["archive"])

            if kwargs["archive"]["chromosome"].str.contains(member).any():  # Get from archive if present.
                # print(kwargs["archive"]["fitness"][kwargs["archive"]["chromosome"] == member].iloc[0])
                result.append(kwargs["archive"]["fitness"][kwargs["archive"]["chromosome"] == member].iloc[0])


            else:
                result.append(fitness_function(*self._chromosome.parameters(member), *args))

        # print(result)
        return result
