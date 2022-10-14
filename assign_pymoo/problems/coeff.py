import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_pymoo.problem import *

__all__ = ['CoefficientProblemBase', 'SmallExpCoeffProblem']


class CoefficientProblemBase(AssignmentProblem):
    """Test problem that allows any connection between n_src and n_tgt nodes, where the objective is calculated from
    the sum of the products of the coefficient of each connection.
    The number of possible connections is given by 2^(n_src*n_tgt)."""

    def __init__(self, encoder):
        self.src_coeff, self.tgt_coeff = self.get_coefficients()
        super().__init__(encoder)

    def get_src_tgt_nodes(self) -> Tuple[List[Node], List[Node]]:
        src_nodes = [Node(min_conn=0, repeated_allowed=False) for _ in self.src_coeff]
        tgt_nodes = [Node(min_conn=0, repeated_allowed=False) for _ in self.tgt_coeff]
        return src_nodes, tgt_nodes

    def _do_evaluate(self, conns: List[Tuple[int, int]], x_aux: Optional[DesignVector]) -> Tuple[List[float], List[float]]:
        coeff_sum = 0.
        for i_src, i_tgt in conns:
            conn_val = self.src_coeff[i_src]*self.tgt_coeff[i_tgt]
            coeff_sum += conn_val
        return [-coeff_sum], []

    def get_coefficients(self) -> Tuple[List[float], List[float]]:
        raise NotImplementedError


class SmallExpCoeffProblem(CoefficientProblemBase):

    # 2^(2*3) = 64 points
    _n_src = 2
    _n_tgt = 3

    def get_coefficients(self) -> Tuple[List[float], List[float]]:
        return self._get_exp_coeff(self._n_src), self._get_exp_coeff(self._n_tgt)

    @staticmethod
    def _get_exp_coeff(n):
        return np.exp(((np.arange(n)+1)/n)-1)


class MediumExpCoeffProblem(SmallExpCoeffProblem):

    # 2^(3*4) = 4096 points
    _n_src = 3
    _n_tgt = 4


if __name__ == '__main__':
    from assign_enc.eager.encodings import *
    from assign_enc.eager.imputation import *
    from assign_enc.lazy.encodings import *
    from assign_enc.lazy.imputation import *
    # enc = DirectMatrixEncoder(FirstImputer())
    enc = LazyDirectMatrixEncoder(LazyFirstImputer())

    # Strange setup like this for profiling
    import timeit

    def _start_comp():
        s = timeit.default_timer()
        p = SmallExpCoeffProblem(enc)
        if isinstance(enc, EagerEncoder):
            print(p.assignment_manager.matrix.shape[0])
            p.eval_points()
        else:
            print(p.get_matrix_count())
        print(f'{timeit.default_timer()-s} sec')

    def _do_real():
        s = timeit.default_timer()
        p = MediumExpCoeffProblem(enc)
        if isinstance(enc, EagerEncoder):
            print(p.assignment_manager.matrix.shape[0])
            p.eval_points()
        else:
            print(p.get_matrix_count())
        print(f'{timeit.default_timer()-s} sec')

    _start_comp()
    _do_real()
