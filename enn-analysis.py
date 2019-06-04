

class test_nenn:
    def __init__(self, dim):
        self.dim = dim

    def fitness(self, x):
        return [sum(x ** 2)]

    def get_bounds(self):
        return ([-1] * self.dim, [1] * self.dim)

    def get_name(self):
        return "Your mother"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)


from pygmo import *
import pygmo as pg

# udp = dtlz(prob_id = 1)


problem = pg.problem(test_nenn(3))

algo = pg.algorithm(pg.bee_colony(gen = 20, limit = 20))
pop = pg.population(problem,10)
pop = algo.evolve(pop)
print(pop.champion_f) #doctest: +SKI

#
#
# pop = population(prob = udp, size = 105)
# # algo = algorithm(moead(gen = 100))
# algo = algorithm(sea(gen = 100))
#
# for i in range(10):
#     pop = algo.evolve(pop)
#     print(udp.p_distance(pop))
#
#
# hv = hypervolume(pop)
# hv.compute(ref_point = [1.,1.,1.])
#
# from matplotlib import pyplot as plt
# udp.plot(pop)
# plt.title("DTLZ1 - MOEAD - GRID - TCHEBYCHEFF")
#
# print(algo)
