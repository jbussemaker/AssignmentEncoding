import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_pymoo.problem import *

__all__ = ['AnalyticalProblemBase', 'AnalyticalCombinationProblem', 'AnalyticalAssignmentProblem',
           'AnalyticalPartitioningProblem', 'AnalyticalDownselectingProblem', 'AnalyticalConnectingProblem',
           'AnalyticalPermutingProblem', 'AnalyticalIterCombinationsProblem',
           'AnalyticalIterCombinationsReplacementProblem']


class AnalyticalProblemBase(AssignmentProblem):
    """Test problem where the objective is calculated from the sum of the products of the coefficient of each
    connection, defined by an exponential function."""

    _invert_f2 = False

    def __init__(self, encoder, n_src: int = 2, n_tgt: int = 3):
        self._n_src = n_src
        self._n_tgt = n_tgt
        self.src_coeff, self.tgt_coeff = self.get_coefficients()
        super().__init__(encoder)

    def get_init_kwargs(self) -> dict:
        return {'n_src': self._n_src, 'n_tgt': self._n_tgt}

    def get_src_tgt_nodes(self) -> Tuple[List[Node], List[Node]]:
        src_nodes = [self._get_node(True, i) for i in range(self.src_coeff.shape[1])]
        tgt_nodes = [self._get_node(False, i) for i in range(self.tgt_coeff.shape[1])]
        return src_nodes, tgt_nodes

    def get_n_obj(self) -> int:
        return 2

    def get_n_valid_design_points(self, n_cont=5) -> int:
        return self.assignment_manager.matrix_gen.count_all_matrices()

    def _do_evaluate(self, conns: List[Tuple[int, int]], x_aux: Optional[DesignVector]) -> Tuple[List[float], List[float]]:
        coeff_sum = np.zeros((2,))
        for i_src, i_tgt in conns:
            coeff_sum += self.src_coeff[:, i_src]*self.tgt_coeff[:, i_tgt]
        coeff_sum[0] = -coeff_sum[0]
        if self._invert_f2:
            coeff_sum[1] = -coeff_sum[1]
        return list(coeff_sum), []

    def get_coefficients(self):
        src_coeff = np.row_stack([self._get_sin_coeff(self._n_src), self._get_cos_coeff(self._n_src)])
        tgt_coeff = np.row_stack([self._get_sin_coeff(self._n_tgt), self._get_cos_coeff(self._n_tgt)])
        return src_coeff, tgt_coeff

    @staticmethod
    def _get_exp_coeff(n):
        return np.exp(((np.arange(n)+1)/n)-1)

    @staticmethod
    def _get_sin_coeff(n):
        return 1-np.sin(np.linspace(.75*np.pi, 2.*np.pi, n))

    @staticmethod
    def _get_cos_coeff(n):
        return np.cos(np.linspace(.25*np.pi, 2.25*np.pi, n))+1

    def _get_node(self, src: bool, idx: int) -> Node:
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class AnalyticalCombinationProblem(AnalyticalProblemBase):
    """Combination pattern: 1 source selecting one of the targets:
    source has 1 connection, targets 0 or 1, no repetitions"""

    def __init__(self, encoder, n_tgt: int = 3):
        super().__init__(encoder, n_src=1, n_tgt=n_tgt)

    def get_init_kwargs(self) -> dict:
        return {'n_tgt': self._n_tgt}

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node([1], repeated_allowed=False) if src else Node([0, 1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_tgt={self._n_tgt})'

    def __str__(self):
        return f'An Comb Prob {self._n_src} -> {self._n_tgt}'


class AnalyticalAssignmentProblem(AnalyticalProblemBase):
    """Assignment pattern: connect any source to any target, optionally with injective/surjective/bijective constraints:
    sources and target have any number of connections, no repetitions;
    - injective: targets have max 1 connection
    - surjective: targets have min 1 connection (note: this is the same as a partitioning problem!)
    - bijective (= injective & surjective): targets have exactly 1 connection"""

    def __init__(self, *args, injective=False, surjective=False, **kwargs):
        self.injective = injective
        self.surjective = surjective
        super().__init__(*args, **kwargs)

    def get_init_kwargs(self) -> dict:
        return {**super().get_init_kwargs(), **{'injective': self.injective, 'surjective': self.surjective}}

    def _get_node(self, src: bool, idx: int) -> Node:
        if src:
            return Node(min_conn=0, repeated_allowed=False)
        if self.injective:
            if self.surjective:
                return Node([1], repeated_allowed=False)
            return Node([0, 1], repeated_allowed=False)
        return Node(min_conn=1 if self.surjective else 0, repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_src={self._n_src}, n_tgt={self._n_tgt}, ' \
               f'injective={self.injective}, surjective={self.surjective})'

    def __str__(self):
        return f'An Assign Prob {self._n_src} -> {self._n_tgt}{"; inj" if self.injective else ""}' \
               f'{"; sur" if self.surjective else ""}'


class AnalyticalPartitioningProblem(AnalyticalProblemBase):
    """Partitioning pattern: sources have any connections, targets 1, no repetitions; if changed into a covering
    partitioning pattern, targets have no max connections"""

    _invert_f2 = True

    def __init__(self, *args, covering=False, **kwargs):
        self.covering = covering
        super().__init__(*args, **kwargs)

    def get_init_kwargs(self) -> dict:
        return {**super().get_init_kwargs(), **{'covering': self.covering}}

    def _get_node(self, src: bool, idx: int) -> Node:
        if src:
            return Node(min_conn=0, repeated_allowed=False)
        return Node(min_conn=1, repeated_allowed=False) if self.covering else Node([1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_src={self._n_src}, n_tgt={self._n_tgt}, ' \
               f'covering={self.covering})'

    def __str__(self):
        return f'An Part Prob {self._n_src} -> {self._n_tgt}{"; cov" if self.covering else ""}'


class AnalyticalDownselectingProblem(AnalyticalProblemBase):
    """Downselecting pattern: 1 source selecting one or more of the targets:
    source has any number connection, targets 0 or 1, no repetitions"""

    def __init__(self, encoder, n_tgt: int = 3):
        super().__init__(encoder, n_src=1, n_tgt=n_tgt)

    def get_init_kwargs(self) -> dict:
        return {'n_tgt': self._n_tgt}

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node(min_conn=0, repeated_allowed=False) if src else Node([0, 1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_tgt={self._n_tgt})'

    def __str__(self):
        return f'An Down Prob {self._n_src} -> {self._n_tgt}'


class AnalyticalConnectingProblem(AnalyticalProblemBase):
    """Connecting pattern: connecting multiple nodes to each other (sources are the same as targets:
    sources and targets have any nr of connections, but cannot be connected to themselves, no repetitions;
    optionally undirected --> only one triangle of the matrix can be connected to"""

    def __init__(self, encoder, n: int = 3, directed=True):
        self._src_nodes = [Node(min_conn=0, repeated_allowed=False) for _ in range(n)]
        self._tgt_nodes = [Node(min_conn=0, repeated_allowed=False) for _ in range(n)]
        self._n = n
        self.directed = directed
        super().__init__(encoder, n_src=n, n_tgt=n)

    def get_init_kwargs(self) -> dict:
        return {'n': self._n, 'directed': self.directed}

    def _get_node(self, src: bool, idx: int) -> Node:
        return self._src_nodes[idx] if src else self._tgt_nodes[idx]

    def get_excluded_edges(self) -> Optional[List[Tuple[Node, Node]]]:
        no_diagonal = [(self._src_nodes[i], self._tgt_nodes[i]) for i in range(self._n)]
        if self.directed:
            return no_diagonal
        no_lower_triangle = [(self._src_nodes[i], self._tgt_nodes[j])
                             for i in range(self._n) for j in range(self._n) if i > j]
        return no_diagonal+no_lower_triangle

    def __repr__(self):
        return f'{self.__class__.__name__}(n={self._n_src}, directed={self.directed})'

    def __str__(self):
        return f'An Conn Prob {self._n_src} -> {self._n_tgt}{"; dir" if self.directed else ""}'


class AnalyticalPermutingProblem(AnalyticalProblemBase):
    """Permutation pattern: sources and targets (same amount) have 1 connection each, no repetitions"""

    def __init__(self, encoder, n: int = 3):
        super().__init__(encoder, n_src=n, n_tgt=n)

    def get_init_kwargs(self) -> dict:
        return {'n': self._n_src}

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node([1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n={self._n_src})'

    def __str__(self):
        return f'An Perm Prob {self._n_src} -> {self._n_tgt}'


class AnalyticalIterCombinationsProblem(AnalyticalProblemBase):
    """Itertools combinations function (select n_take elements from n_tgt targets):
    1 source has n_take connections to n_tgt targets, no repetition"""

    _invert_f2 = True

    def __init__(self, encoder, n_take: int = 2, n_tgt: int = 3):
        self._n_take = min(n_take, n_tgt)
        super().__init__(encoder, n_src=1, n_tgt=n_tgt)

    def get_init_kwargs(self) -> dict:
        return {'n_take': self._n_take, 'n_tgt': self._n_tgt}

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node([self._n_take], repeated_allowed=False) if src else Node([0, 1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_take={self._n_take}, n_tgt={self._n_tgt})'

    def __str__(self):
        return f'An Iter Comb Prob {self._n_take} from {self._n_tgt}'


class AnalyticalIterCombinationsReplacementProblem(AnalyticalIterCombinationsProblem):
    """Itertools combinations_with_replacement function (select n_take elements from n_tgt targets):
    1 source has n_take connections to n_tgt targets, repetition allowed"""

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node([self._n_take]) if src else Node(min_conn=0)

    def __str__(self):
        return f'An Iter Comb Repl Prob {self._n_take} from {self._n_tgt}'


if __name__ == '__main__':
    from assign_enc.eager.encodings import *
    from assign_enc.eager.imputation import *
    from assign_enc.lazy.encodings import *
    from assign_enc.lazy.imputation import *
    enc = DirectMatrixEncoder(FirstImputer())
    # enc = OneVarEncoder(FirstImputer())
    # enc = LazyDirectMatrixEncoder(LazyFirstImputer())

    # Strange setup like this for profiling
    import timeit

    def _start_comp():
        s = timeit.default_timer()
        p = AnalyticalAssignmentProblem(enc)
        if isinstance(enc, EagerEncoder):
            print(p.assignment_manager.matrix.shape[0])
            # p.eval_points()
        else:
            print(p.get_matrix_count())
        print(f'{timeit.default_timer()-s} sec')

        print(f'Imputation ratio: {p.get_imputation_ratio(n_sample=None)}')
        print(f'Information error: {p.get_information_error()[0, 0]}')

    def _do_real():
        s = timeit.default_timer()
        p = AnalyticalAssignmentProblem(enc, n_src=3, n_tgt=4)
        if isinstance(enc, EagerEncoder):
            print(p.assignment_manager.matrix.shape[0])
            # p.eval_points()
        else:
            print(p.get_matrix_count())
        print(f'{timeit.default_timer()-s} sec')

        print(f'Imputation ratio: {p.get_imputation_ratio(n_sample=None)}')
        print(f'Information error: {p.get_information_error()[0, 0]}')

    # _start_comp()
    # _do_real(), exit()

    from assign_enc.encoder_registry import *
    from assign_pymoo.metrics_compare import *
    # p = AnalyticalCombinationProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER())  # Very high imputation ratios
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER(), injective=True, surjective=True)
    p = AnalyticalPartitioningProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalPartitioningProblem(LazyAmountFirstEncoder(DEFAULT_LAZY_IMPUTER(), FlatLazyAmountEncoder(), FlatLazyConnectionEncoder()), n_src=2, n_tgt=4, covering=True)
    # p = AnalyticalDownselectingProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalConnectingProblem(DEFAULT_EAGER_ENCODER())  # Low information errors
    # p = AnalyticalPermutingProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalIterCombinationsProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalIterCombinationsReplacementProblem(DEFAULT_EAGER_ENCODER(), n_take=3, n_tgt=3)  # Low inf err
    # p.plot_pf(show_approx_f_range=True), exit()
    enc = []
    enc += [e(DEFAULT_EAGER_IMPUTER()) for e in EAGER_ENCODERS]
    enc += [e(DEFAULT_LAZY_IMPUTER()) for e in LAZY_ENCODERS]
    # MetricsComparer().compare_encoders(p, enc)
    MetricsComparer().compare_encoders(p, enc, inf_idx=True)
    # MetricsComparer(n_samples=50, n_leave_out=30).check_information_corr(p, enc)
