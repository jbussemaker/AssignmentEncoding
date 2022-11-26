import os
import re
import pickle
import hashlib
import numpy as np
from typing import *
import concurrent.futures
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_enc.selector import *
from assign_pymoo.sampling import *
from assign_enc.lazy_encoding import *
from assign_enc.assignment_manager import *

from pymoo.optimize import minimize
from pymoo.core.variable import Real
from pymoo.core.repair import Repair
from pymoo.core.problem import Problem
from pymoo.core.variable import Integer
from pymoo.problems.multi.zdt import ZDT1
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.visualization.scatter import Scatter
from pymoo.core.initialization import Initialization
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.algorithms.moo.nsga2 import NSGA2, calc_crowding_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


__all__ = ['AssignmentProblem', 'AssignmentRepair', 'CachedParetoFrontMixin']


class CachedParetoFrontMixin(Problem):
    """Mixin to calculate the Pareto front once by simply running the problem a couple of times using NSGA2. Stores the
    results based on the repr of the main class, so make sure that one is set."""

    def reset_pf_cache(self):
        cache_path = self._pf_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def get_repair(self):
        pass

    def _calc_pareto_front(self, *_, pop_size=200, n_gen=20, n_repeat=10, n_pts_keep=50, **__):
        cache_path = self._pf_cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as fp:
                return pickle.load(fp)

        n = 1
        xl, xu = self.bounds()
        for i, var in enumerate(self.vars):
            if isinstance(var, Real):
                n = None
                break
            n *= int(xu[i]-xl[i]+1)
        if n is not None and n < pop_size*n_gen*n_repeat:
            sampling = RepairedExhaustiveSampling(repair=self.get_repair())
            pop = sampling.do(self, n)
            Evaluator().eval(self, pop)

            pop = DefaultDuplicateElimination().do(pop)
            pf = pop.get('F')
            i_non_dom = NonDominatedSorting().do(pf, only_non_dominated_front=True)
            pf = pf[i_non_dom, :]

        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._run_minimize, pop_size, n_gen, i, n_repeat)
                           for i in range(n_repeat)]
                concurrent.futures.wait(futures)

                pf = None
                for i in range(n_repeat):
                    res = futures[i].result()
                    if pf is None:
                        pf = res.F
                    else:
                        pf_merged = np.row_stack([pf, res.F])
                        i_non_dom = NonDominatedSorting().do(pf_merged, only_non_dominated_front=True)
                        pf = pf_merged[i_non_dom, :]

        pf = np.unique(pf, axis=0)
        if n_pts_keep is not None and pf.shape[0] > n_pts_keep:
            for _ in range(pf.shape[0]-n_pts_keep):
                crowding_of_front = calc_crowding_distance(pf)
                i_max_crowding = np.argsort(crowding_of_front)[1:]
                pf = pf[i_max_crowding, :]

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as fp:
            pickle.dump(pf, fp)
        return pf

    def _run_minimize(self, pop_size, n_gen, i, n):
        print('Running PF discovery %d/%d (%d pop, %d gen)' % (i+1, n, pop_size, n_gen))
        return minimize(self, NSGA2(pop_size=pop_size), termination=('n_gen', n_gen))

    def plot_pf(self: Union[Problem, 'CachedParetoFrontMixin'], show_approx_f_range=False, **kwargs):
        scatter = Scatter()
        if show_approx_f_range:
            scatter.add(self.get_approx_f_range(), s=.1, color='white')

            pop = Initialization(LatinHypercubeSampling()).do(self, 100)
            Evaluator().eval(self, pop)
            scatter.add(pop.get('F'), s=5)

        scatter.add(self.pareto_front(**kwargs))
        scatter.show()

    def get_approx_f_range(self, n_sample=1000):
        pop = Initialization(LatinHypercubeSampling()).do(self, n_sample)
        Evaluator().eval(self, pop)
        f = pop.get('F')
        f_max = np.max(f, axis=0)
        f_min = np.min(f, axis=0)
        return np.array([f_min, f_max])

    def _pf_cache_path(self):
        class_str = repr(self)
        if class_str.startswith('<'):
            class_str = self.__class__.__name__
        class_str = re.sub('[^0-9a-z]', '_', class_str.lower().strip())

        if len(class_str) > 20:
            class_str = hashlib.md5(class_str.encode('utf-8')).hexdigest()[:20]

        return os.path.join(os.path.dirname(__file__), '.pf_cache', class_str+'.pkl')


class AssignmentProblem(CachedParetoFrontMixin, Problem):
    """class representing an assignment optimization problem."""

    def __init__(self, encoder: Encoder = None, **_):
        src, tgt = self.get_src_tgt_nodes()
        excluded = self.get_excluded_edges()
        existence_patterns = self.get_existence_patterns()
        if encoder is None:
            assignment_manager = EncoderSelector(
                src, tgt, excluded=excluded, existence_patterns=existence_patterns).get_best_assignment_manager()
        elif isinstance(encoder, (LazyEncoder, EagerEncoder)):
            args, kwargs = (src, tgt, encoder), {'excluded': excluded, 'existence_patterns': existence_patterns}
            cls = LazyAssignmentManager if isinstance(encoder, LazyEncoder) else AssignmentManager
            assignment_manager = cls(*args, **kwargs)
        else:
            raise RuntimeError(f'Unknown encoder type: {encoder}')
        self.assignment_manager = assignment_manager
        design_vars = assignment_manager.design_vars

        aux_des_vars = self.get_aux_des_vars() or []
        self.n_aux = len(aux_des_vars)
        design_vars = aux_des_vars+design_vars

        n_var = len(design_vars)
        var_types = [Integer(bounds=(0, dv.n_opts-1)) for dv in design_vars]
        xl = np.zeros((n_var,), dtype=int)
        xu = np.array([dv.n_opts-1 for dv in design_vars])

        n_obj, n_con = self.get_n_obj(), self.get_n_con()+1
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_con, xl=xl, xu=xu, vars=var_types)

    def reset_encoder_selector_cache(self):
        src, tgt = self.get_src_tgt_nodes()
        excluded = self.get_excluded_edges()
        existence_patterns = self.get_existence_patterns()
        EncoderSelector(src, tgt, excluded=excluded, existence_patterns=existence_patterns).reset_cache()

    def get_for_encoder(self, encoder: Encoder):
        return self.__class__(encoder, **self.get_init_kwargs())

    def get_matrix_count(self):
        src, tgt = self.get_src_tgt_nodes()
        excluded = self.get_excluded_edges()
        existence_patterns = self.get_existence_patterns()
        matrix_gen = AggregateAssignmentMatrixGenerator(
            src, tgt, excluded=excluded, existence_patterns=existence_patterns)
        return matrix_gen.count_all_matrices()

    def get_repair(self):
        return AssignmentRepair()

    def correct_x(self, x: np.ndarray) -> np.ndarray:
        n_aux = self.n_aux
        x_corr = x.copy().astype(np.int)
        for i_dv in range(x_corr.shape[0]):
            existence = None
            if n_aux > 0:
                x_corr[i_dv, :n_aux], existence, _ = self.correct_x_aux(x_corr[i_dv, :n_aux])
            x_corr[i_dv, n_aux:] = self.assignment_manager.correct_vector(x_corr[i_dv, n_aux:], existence=existence)
        return x_corr

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.round(x.astype(np.float64)).astype(np.int)
        n = x.shape[0]
        x_out = np.empty((n, self.n_var))
        f_out = np.empty((n, self.n_obj))
        g_out = np.zeros((n, self.n_constr))

        n_aux = self.n_aux
        for i in range(n):
            x_aux, existence = None, None
            if n_aux > 0:
                x_aux, existence, is_violated = self.correct_x_aux(x[i, :n_aux])
                x_out[i, :n_aux] = x_aux
                if is_violated:
                    f_out[i, :] = 0
                    g_out[i, 0] = 1
                    continue

            x_out[i, n_aux:], conn_idx = self.assignment_manager.get_conn_idx(x[i, n_aux:], existence=existence)
            if conn_idx is None:
                f_out[i, :] = 0
                g_out[i, 0] = 1
            else:
                f_out[i, :], g_out[i, 1:] = self._do_evaluate(conn_idx, x_aux=x_aux)

        out['X'] = x_out
        out['F'] = f_out
        out['G'] = g_out

    def get_n_design_points(self, n_cont=5) -> int:
        n = 1
        xl, xu = self.bounds()
        for i, var in enumerate(self.vars):
            n *= n_cont if isinstance(var, Real) else int(xu[i]-xl[i]+1)
        return n

    def get_n_valid_design_points(self, n_cont=5) -> int:
        """Implement if the number of valid design points can be calculated analytically"""

    def get_init_sampler(self, lhs=True, **kwargs):
        return get_init_sampler(repair=self.get_repair(), lhs=lhs, **kwargs)

    def sample_points(self, n=None, n_cont=5, remove_duplicates=True, lhs=True) -> Population:
        if n is not None:
            n_des_points = self.get_n_design_points(n_cont=n_cont)
            if n > n_des_points:
                n = None

        repair = self.get_repair()
        if n is None:
            sampler = RepairedExhaustiveSampling(repair=repair, n_cont=n_cont, remove_duplicates=remove_duplicates)
        elif lhs:
            sampler = RepairedLatinHypercubeSampling(repair=repair)
        else:
            sampler = RepairedRandomSampling(repair=repair)

        return sampler.do(self, n or 100)

    def eval_points(self, n=None) -> Population:
        pop = self.sample_points(n=n)
        print(f'Points sampled (n={n!r}): {len(pop)}')
        return Evaluator().eval(self, pop)

    def plot_points(self, pop: Population = None, n=None):
        if pop is None:
            pop = self.eval_points(n=n)
        Scatter().add(pop.get('F')).show()

    def get_information_error(self, n_samples: int = None, n_leave_out: int = None, **kwargs) -> np.ndarray:
        from assign_pymoo.information_content import InformationContentAnalyzer
        if n_samples is None:
            n_samples = self.n_var*5
            n_leave_out = 10
        return InformationContentAnalyzer(self, **kwargs).get_information_error(n_samples, n_leave_out=n_leave_out)

    def get_imputation_ratio(self, n_sample: Optional[int] = 10000):
        # return self.assignment_manager.get_imputation_ratio()

        # Check if analytically available
        n_cont = 5
        n_real = self.get_n_valid_design_points(n_cont=n_cont)
        if n_real is not None:
            n = self.get_n_design_points(n_cont=n_cont)
            return n/n_real

        # Determining the imputation ratio based on sampling (if not exhaustive) can be very inaccurate due to
        # non-uniform distribution of imputed design points
        pop = self.sample_points(n=n_sample, remove_duplicates=False, lhs=False)
        x = pop.get('X')
        n = x.shape[0]
        n_unique = np.unique(x, axis=0).shape[0]
        return n/n_unique

    def get_information_index(self) -> float:
        xl, xu = self.bounds()
        n_opts = [int(xu[i]-xl[i]+1) for i, var in enumerate(self.vars) if isinstance(var, Integer)]
        return Encoder.calc_information_index(n_opts)

    def name(self):
        return f'{self!s} / {self.assignment_manager.encoder!s}'

    # !! IMPLEMENT BELOW THIS LINE !! #

    def get_init_kwargs(self) -> dict:
        raise NotImplementedError

    def get_n_obj(self) -> int:
        return 1

    def get_n_con(self) -> int:
        return 0

    def get_aux_des_vars(self) -> Optional[List[DiscreteDV]]:
        """Additional discrete design variables"""
        pass

    def get_src_tgt_nodes(self) -> Tuple[List[Node], List[Node]]:
        raise NotImplementedError

    def get_excluded_edges(self) -> Optional[List[Tuple[Node, Node]]]:
        pass

    def get_existence_patterns(self) -> Optional[NodeExistencePatterns]:
        pass

    def correct_x_aux(self, x_aux: DesignVector) -> Tuple[DesignVector, Optional[NodeExistence], bool]:
        """Correct auxiliary design vector, return imputed design vector, optional NodeExistence, and flag whether the
        design is invalid"""
        raise RuntimeError

    def _do_evaluate(self, conns: List[Tuple[int, int]], x_aux: Optional[DesignVector]) -> Tuple[List[float], List[float]]:
        """Returns [objectives, constraints]"""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class AssignmentRepair(Repair):

    def do(self, problem, pop, **kwargs):
        is_array = not isinstance(pop, Population)
        x = pop if is_array else pop.get('X')

        x = np.round(x.astype(np.float64)).astype(np.int)
        if isinstance(problem, AssignmentProblem):
            x = problem.correct_x(x)

        if is_array:
            return x
        pop.set('X', x)
        return pop


class ZDT1Calc(CachedParetoFrontMixin, ZDT1):
    pass


if __name__ == '__main__':
    ZDT1Calc().plot_pf()

