import itertools
import numpy as np
from pymoo.core.repair import Repair
from pymoo.core.variable import Real
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.population import Population
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.operators.sampling.lhs import LatinHypercubeSampling, sampling_lhs_unit

__all__ = ['RepairedExhaustiveSampling', 'RepairedLatinHypercubeSampling']


class RepairedExhaustiveSampling(Sampling):

    def __init__(self, repair: Repair = None, n_cont=5):
        super().__init__()
        self._repair = repair
        self._n_cont = n_cont

    def _do(self, problem: Problem, n_samples, **kwargs):
        xl, xu = problem.bounds()
        is_cont = np.ones((len(xl),), dtype=bool)
        if problem.vars is not None:
            for i, var in enumerate(problem.vars):
                if not isinstance(var, Real):
                    is_cont[i] = False

        opt_values = [np.linspace(xl[i], xu[i], self._n_cont) if is_cont[i] else np.arange(xl[i], xu[i]+1)
                      for i in range(len(xl))]
        x = np.array([np.array(dv) for dv in itertools.product(*opt_values)])

        pop = Population.new(X=x)
        pop = self._repair.do(problem, pop)
        pop = DefaultDuplicateElimination().do(pop)
        return pop.get('X')


class RepairedLatinHypercubeSampling(LatinHypercubeSampling):
    """Latin hypercube sampling only returning repaired samples."""

    def __init__(self, repair: Repair = None, **kwargs):
        super().__init__(**kwargs)
        self._repair = repair

    def _do(self, problem: Problem, n_samples, **kwargs):
        if self._repair is None:
            return super()._do(problem, n_samples, **kwargs)

        xl, xu = problem.bounds()

        X = sampling_lhs_unit(n_samples, problem.n_var, smooth=self.smooth)
        X = self.repair_x(problem, X, xl, xu)

        if self.criterion is not None:
            score = self.criterion(X)
            for j in range(1, self.iterations):

                _X = sampling_lhs_unit(n_samples, problem.n_var, smooth=self.smooth)
                _X = self.repair_x(problem, _X, xl, xu)
                _score = self.criterion(_X)

                if _score > score:
                    X, score = _X, _score

        return xl + X * (xu - xl)

    def repair_x(self, problem, x, xl, xu):
        x_abs = x*(xu-xl)+xl
        x_abs = self._repair.do(problem, Population.new(X=x_abs)).get("X")
        return (x_abs-xl)/(xu-xl)
