import numpy as np
from CNN_Class import CNN, random_split, CustomDataset
import pygmo as pg

NORMALIZE = True
IMAGE_PATH = 'database/'
dataset = CustomDataset(image_path=IMAGE_PATH, normalise=NORMALIZE, train=True)
lengths = [10000, 10778]  # train data and test data
train_dataset, test_dataset = random_split(dataset, lengths)  # 20778


def argument_input_interface(
        n_conv,
        kernel_conv,
        stride_conv,
        kernel_pool,
        stride_pool,
        n_layers,
        dim1=(6, 32),
        dim2=(120, 40)
):
    _dim1 = np.linspace(dim1[0], dim1[1], int(n_conv)).astype(int)
    _dim2 = np.linspace(dim2[0], dim2[1], int(n_conv)).astype(int)
    _kernel_conv = [int(kernel_conv)] * int(n_conv)
    _stride_conv = [int(stride_conv)] * int(n_conv)
    _kernel_pool = [int(kernel_pool)] * int(n_conv)
    _stride_pool = [int(stride_pool)] * int(n_conv)
    print(int(n_conv), list(_dim1), _kernel_conv, _stride_conv, _kernel_pool, _stride_pool, int(n_layers), list(_dim2))
    return int(n_conv), list(_dim1), _kernel_conv, _stride_conv, _kernel_pool, _stride_pool, int(n_layers), list(_dim2)


class NeuroEvolutionaryNetwork:
    def __init__(self):
        self.network_class = CNN

    def fitness(self, x):
        session = self.network_class(
            train_dataset,
            test_dataset,
            *argument_input_interface(*x)
        )
        return [np.min(session.losslst), session.realtime]

    def get_nobj(self):
        return 2

    def get_name(self):
        return "NeuroEvolutionary Network"

    def get_bounds(self):
        return (
            # n_conv . kernel_conv . stride_conv . kernel_pool . stride_pool . n_layers
            [3,        3,            1,            2,            2,            3],
            [3,        3,            1,            2,            2,            3]
        )


if __name__ == "__main__":
    problem = pg.problem(NeuroEvolutionaryNetwork())
    pop = pg.population(problem, size=20)
    algo = pg.algorithm(pg.nsga2(gen=40))
    pop = algo.evolve(pop)
    fits, vectors = pop.get_f(), pop.get_x()
    # extract and print non-dominated fronts
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(fits)
