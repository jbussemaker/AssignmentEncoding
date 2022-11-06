from assign_pymoo.sbo import *
from assign_pymoo.sampling import *
from pymoo.algorithms.moo.nsga2 import NSGA2
from assign_pymoo.problem import AssignmentProblem

__all__ = ['get_ga_algo', 'get_sbo_algo']


def get_ga_algo(problem: AssignmentProblem, pop_size=100):
    """NSGA2 (a multi-objective genetic algorithm) running a fixed number of generations"""
    return NSGA2(pop_size=pop_size, sampling=RepairedLatinHypercubeSampling(problem.get_repair()))


def get_sbo_algo(problem: AssignmentProblem, init_size=100):
    """Kriging SBO algorithm with a given initial DOE size (init_size) and a number of infill points (n_infill)"""
    return get_sbo_krg(use_mvpf=True, init_size=init_size, repair=problem.get_repair())
    # return get_sbo_rbf(init_size=init_size, repair=problem.get_repair())
