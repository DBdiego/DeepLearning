import numpy as np
from CNN_Class import CNN, random_split, CustomDataset
import pygmo as pg
import torch

NORMALIZE = True
IMAGE_PATH = 'database/'
RATIO_TRAINING = 0.1
RATIO_DATA = 0.1
MAX_DATA = RATIO_DATA*41556


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
        dim1=32,
        dim2=120
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
        final_dim = int((final_dim - kernel_pool + 2 * 0) / stride_pool + 1)
        #
        bad = True if (final_dim == 0) else False
        #
        if bad:
            raise ArithmeticError
    # print(int(n_conv), list(_dim1), _kernel_conv, _stride_conv, _kernel_pool, _stride_pool, int(n_layers), list(_dim2))
    return int(n_conv), list(_dim1), _kernel_conv, _stride_conv, _kernel_pool, _stride_pool, int(n_layers), list(_dim2)

class NeuroEvolutionaryNetwork:
    def __init__(self):
        self.network_class = CNN

    def fitness(self, x):
        try:
            values = argument_input_interface(*x)

            session = self.network_class(
                train_dataset,
                test_dataset,
                *values
            )
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
            [2, 2, 1, 1, 2, 4],
            [8, 3, 2, 1, 3, 15]
        )


if __name__ == "__main__":
    print("Available GPU's: ", torch.cuda.device_count(), '\n')
    problem = pg.problem(NeuroEvolutionaryNetwork())
    pop = pg.population(problem, size=20)
    algo = pg.algorithm(pg.nsga2(gen=20))
    pop = algo.evolve(pop)
    fits, vectors = pop.get_f(), pop.get_x()
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)

    np.savetxt('results.txt', fits, delimiter=';')
    
