import numpy as np
from CNN_Class import CNN, random_split, CustomDataset
import pygmo as pg
import torch
import time

NORMALIZE = True
IMAGE_PATH = 'database/'
RATIO_TRAINING = 0.1
RATIO_DATA = 0.1
MAX_DATA = RATIO_DATA*41556

print('Running ENN.py\n')

print('Importing data: ...')
dataset = CustomDataset(image_path=IMAGE_PATH, normalise=NORMALIZE, maxx=MAX_DATA, train=True)
print('Importing data: DONE\n')

I = int(RATIO_TRAINING * len(dataset))
lengths = [len(dataset) - I, I]  # train data and test data
train_dataset, test_dataset = random_split(dataset, lengths)  # 20778x2
# train_dataset = train_dataset[0::3]
# test_dataset = test_dataset[0::3]


def argument_input_interface(
        n_conv,
        kernel_conv,
        stride_conv,
        kernel_pool,
        stride_pool,
        n_layers,
        dim1=400,
        dim2=200
):
    _dim1 = np.linspace(3, dim1, int(n_conv + 1)).astype(int)
    _dim1 = np.delete(_dim1, 0)
    _dim2 = np.linspace(dim2, 40, int(n_layers)).astype(int)
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
    # print(int(n_conv), list(_dim1), _kernel_conv, _stride_conv, _kernel_pool, _stride_pool, int(n_layers), list(_dim2))
    return int(n_conv), list(_dim1), _kernel_conv, _stride_conv, _kernel_pool, _stride_pool, int(n_layers), list(_dim2)


saved_data = []
if torch.cuda.is_available():
    num_avail_gpus = torch.cuda.device_count()
else:
    num_avail_gpus = 1
    
class NeuroEvolutionaryNetwork:
    def __init__(self):
        self.network_class = CNN

    def fitness(self, x):
        try:
            values = argument_input_interface(*x)
            
            gpu_index = np.random.randint(0, num_avail_gpus)
            
            session = self.network_class(gpu_index, 
                train_dataset,
                test_dataset,
                *values
            )
            saved_data.append([100 - np.max(session.accuracy), session.realtime])
            return [100 - np.max(session.accuracy), session.realtime]

        except ZeroDivisionError:
            return [100, 1000]

        except ArithmeticError:
            return [100, 1000]

        except ValueError:
            return [100, 1000]

    def get_nobj(self):
        return 2

    def get_name(self):
        return "NeuroEvolutionary Network"

    def get_bounds(self):
        return (
            # n_conv . kernel_conv . stride_conv . kernel_pool . stride_pool . n_layers
            [2, 2, 1, 1, 2,  4],
            [8, 3, 2, 1, 3, 15]
        )


if __name__ == "__main__":
    
    print("Available GPU's: ", torch.cuda.device_count(), '\n')

    start_time_prob = time.time()
    print('Problem Definition: ...')
    problem = pg.problem(NeuroEvolutionaryNetwork())
    print(f'Problem Definition: DONE ({round(time.time() - start_time_prob, 5)}s)\n')

    start_time_pop = time.time()
    print('Generating Population: ...')
    pop = pg.population(problem, size=8)
    print(f'Generating Population: DONE ({round(time.time() - start_time_pop, 5)}s)\n')

    start_time_algo = time.time()
    print('Defining Algorithm: ...')
    algo = pg.algorithm(pg.nsga2(gen=2))
    print(f'Defining Algorithm: DONE ({round(time.time() - start_time_algo, 5)}s)\n')

    start_time_archi = time.time()
    print('Creating Archipelago: ...')
    #archi = pg.archipelago(n=2,algo=algo, prob=problem, pop_size=5, udi=pg.ipyparallel_island())
    print(f'Creating Archipelago: DONE ({round(time.time() - start_time_archi, 5)}s)\n')
    
    print('\n===============================================')
    #print(archi)
    print('===============================================\n')

    start_time_evovle = time.time()
    print('Evolving Population: ...')
    pop = algo.evolve(pop)
    #pop = archi.evolve()
    print(f'Evolving Population: DONE ({round(time.time() - start_time_evovle, 5)}s)\n')

    start_time_results = time.time()
    print('Gathering results: ...')
    print(saved_data)
    #fits, vectors = pop.get_f(), pop.get_x()
    #ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
    print(f'Gathering results: DONE ({round(time.time() - start_time_results, 5)}s)\n')

    np.savetxt('results.txt', np.array(saved_data), delimiter=';')












    
