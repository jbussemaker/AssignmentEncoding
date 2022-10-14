import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_pymoo.sampling import *
from assign_enc.lazy_encoding import *
from assign_enc.assignment_manager import *

from pymoo.core.repair import Repair
from pymoo.core.problem import Problem
from pymoo.core.variable import Integer
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.visualization.scatter import Scatter

__all__ = ['AssignmentProblem', 'AssignmentRepair']


class AssignmentProblem(Problem):
    """class representing an assignment optimization problem."""

    def __init__(self, encoder: Union[EagerEncoder, LazyEncoder]):
        src, tgt = self.get_src_tgt_nodes()
        excluded = self.get_excluded_edges()
        if isinstance(encoder, LazyEncoder):
            assignment_manager = LazyAssignmentManager(src, tgt, encoder, excluded=excluded)
        else:
            assignment_manager = AssignmentManager(src, tgt, encoder, excluded=excluded)
        self.assignment_manager = assignment_manager
        design_vars = assignment_manager.design_vars

        aux_des_vars = self.get_aux_des_vars() or []
        self.n_aux = len(aux_des_vars) > 0
        design_vars = aux_des_vars+design_vars

        n_var = len(design_vars)
        var_types = [Integer(bounds=(0, dv.n_opts-1)) for dv in design_vars]
        xl = np.zeros((n_var,), dtype=int)
        xu = np.array([dv.n_opts-1 for dv in design_vars])

        n_obj, n_con = self.get_n_obj(), self.get_n_con()
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_con, xl=xl, xu=xu, vars=var_types)

    def get_matrix_count(self):
        src, tgt = self.get_src_tgt_nodes()
        excluded = self.get_excluded_edges()
        matrix_gen = AggregateAssignmentMatrixGenerator(src, tgt, excluded=excluded)
        return matrix_gen.count_all_matrices()

    def get_repair(self):
        return AssignmentRepair()

    def correct_x(self, x: np.ndarray) -> np.ndarray:
        n_aux = self.n_aux
        x_corr = x.copy().astype(np.int)
        for i_dv in range(x_corr.shape[0]):
            src_exists, tgt_exists = None, None
            if n_aux > 0:
                x_corr[i_dv, :n_aux], src_exists, tgt_exists = self.correct_x_aux(x_corr[i_dv, :n_aux])
            x_corr[i_dv, n_aux:] = self.assignment_manager.correct_vector(
                x_corr[i_dv, n_aux:], src_exists=src_exists, tgt_exists=tgt_exists)
        return x_corr

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.round(x.astype(np.float64)).astype(np.int)
        n = x.shape[0]
        x_out = np.empty((n, self.n_var))
        f_out = np.empty((n, self.n_obj))
        has_g = self.n_constr > 0
        g_out = np.empty((n, self.n_constr)) if has_g else None

        n_aux = self.n_aux
        for i in range(n):
            x_aux, src_exists, tgt_exists = None, None, None
            if n_aux > 0:
                x_aux, src_exists, tgt_exists = self.correct_x_aux(x[i, :n_aux])
                x_out[i, :n_aux] = x_aux

            x_out[i, n_aux:], conn_idx = self.assignment_manager.get_conn_idx(
                x[i, n_aux:], src_exists=src_exists, tgt_exists=tgt_exists)
            f_out[i, :], g_i = self._do_evaluate(conn_idx, x_aux=x_aux)
            if has_g:
                g_out[i, :] = g_i

        out['X'] = x_out
        out['F'] = f_out
        if has_g:
            out['G'] = g_out

    def eval_points(self, n=None) -> Population:
        repair = self.get_repair()
        sampler = RepairedExhaustiveSampling(repair=repair) \
            if n is None else RepairedLatinHypercubeSampling(repair=repair)
        pop = sampler.do(self, n or 100)
        print(f'Points sampled (n={n!r}): {len(pop)}')
        return Evaluator().eval(self, pop)

    def plot_points(self, pop: Population = None, n=None):
        if pop is None:
            pop = self.eval_points(n=n)
        Scatter().add(pop.get('F')).show()

    # !! IMPLEMENT BELOW THIS LINE !! #

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

    def correct_x_aux(self, x_aux: DesignVector) -> Tuple[DesignVector, Optional[List[bool]], Optional[List[bool]]]:
        """Correct auxiliary design vector and additionally return whether src or tgt nodes exist"""
        raise RuntimeError

    def _do_evaluate(self, conns: List[Tuple[int, int]], x_aux: Optional[DesignVector]) -> Tuple[List[float], List[float]]:
        """Returns [objectives, constraints]"""
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
