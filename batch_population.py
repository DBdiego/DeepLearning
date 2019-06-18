from external.sga.population import *
from paralleliser import fitness_func 
import numpy as np


def argument_input_interface(
        n_conv,
        kernel_conv,
        #stride_conv,
        kernel_pool,
        stride_pool,
        n_layers,
        dim1=300,
        dim2=3000
):
    stride_conv = 1
    _dim1 = np.linspace(3, dim1, int(n_conv + 1)).astype(int)
    _dim1 = np.delete(_dim1, 0)
    _dim2 = np.linspace(dim2, 5, int(n_layers)).astype(int)
    _kernel_conv = [int(kernel_conv)] * int(n_conv)
    _stride_conv = [int(stride_conv)] * int(n_conv)
    _kernel_pool = [int(kernel_pool)] * int(n_conv)
    _stride_pool = [int(stride_pool)] * int(n_conv)

    # !!!!!!! final_dim has to be checked that it does not reduce to zero
    # Note that if the stride of the convolution and pooling is held at (1,2), then most likely final_dim will not
    # reduce to zero unless n_conv exceeds 4!

    final_dim = 224
    for i in range(int(n_conv)):
        pad = int(kernel_conv / 2)
        final_dim = int((final_dim - kernel_conv + 2 * pad) / stride_conv + 1)
        final_dim = int((final_dim - kernel_pool + 2 *   0) / stride_pool + 1)
        #
        bad = True if (final_dim == 0) else False
        #
        if bad:
            raise ArithmeticError

    return int(n_conv), list(_dim1), _kernel_conv, _stride_conv, _kernel_pool, _stride_pool, int(n_layers), list(_dim2)


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

        if len(kwargs['archive'].index.tolist()) > 0:
            print(np.array(kwargs['archive'].index.tolist())/len(result))
            generation_index = list(np.array(kwargs['archive'].index.tolist())%len(result)).count(0)
        else:
            generation_index = 0
            
        for index, member in enumerate(self._population):
            
            # If already computed previously
            if kwargs["archive"]["chromosome"].str.contains(member).any():
                result[index] = kwargs["archive"]["fitness"][kwargs["archive"]["chromosome"] == member].iloc[0]


            else:
                try:
                    print(*self._chromosome.parameters(member))
                    genome = argument_input_interface(*self._chromosome.parameters(member))
                    genomes.append(genome)
                    
                except ArithmeticError:
                    result[index] = 0                
        
        result = fitness_func(genomes, generation_index, kwargs['train_dataset'], kwargs['test_dataset'], result)
        print(result)
        return result








