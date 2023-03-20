import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_pymoo.problem import *
from assign_enc.lazy_encoding import *
from assign_enc.patterns.patterns import *
from assign_enc.patterns.encoder import PatternEncoderBase

__all__ = ['AnalyticalProblemBase', 'AnalyticalCombinationProblem', 'AnalyticalAssignmentProblem',
           'AnalyticalPartitioningProblem', 'AnalyticalDownselectingProblem', 'AnalyticalConnectingProblem',
           'AnalyticalPermutingProblem', 'AnalyticalUnorderedNonReplaceCombiningProblem',
           'AnalyticalUnorderedCombiningProblem']


class AnalyticalProblemBase(AssignmentProblem):
    """Test problem where the objective is calculated from the sum of the products of the coefficient of each
    connection, defined by an exponential function."""

    def __init__(self, encoder=None, n_src: int = 2, n_tgt: int = 3):
        self._n_src = n_src
        self._n_tgt = n_tgt
        self.coeff = self.get_coefficients(self._n_src, self._n_tgt)
        self._invert_f2 = False
        self._skew_f2 = False
        super().__init__(encoder)

    def get_init_kwargs(self) -> dict:
        return {'n_src': self._n_src, 'n_tgt': self._n_tgt}

    def get_src_tgt_nodes(self) -> Tuple[List[Node], List[Node]]:
        src_nodes = [self._get_node(True, i) for i in range(self.coeff.shape[0])]
        tgt_nodes = [self._get_node(False, i) for i in range(self.coeff.shape[1])]
        return src_nodes, tgt_nodes

    def get_n_obj(self) -> int:
        return 2

    def get_n_valid_design_points(self, n_cont=5) -> int:
        return self.assignment_manager.matrix_gen.count_all_matrices(max_by_existence=False)

    def _do_evaluate(self, conns: List[Tuple[int, int]], x_aux: Optional[DesignVector]) -> Tuple[List[float], List[float]]:
        f = np.zeros((2,))
        for i_src, i_tgt in conns:
            f += self.coeff[i_src, i_tgt, :]
        if self._invert_f2:
            f[1] = -f[1]
        if self._skew_f2:
            f[1] -= .25*f[0]
            f[0] -= .25*f[1]
        if self.n_obj == 1:
            return list(f)[:1], []
        return list(f), []

    @staticmethod
    def get_coefficients(n_src, n_tgt):
        coeff = np.ones((n_src, n_tgt, 2))
        for i in range(n_src):
            coeff[i, :, 0] = np.sin(np.linspace(0, 3*np.pi, n_tgt) + .25*np.pi*i)-1
            coeff[i, :, 1] = np.cos(np.linspace(-np.pi, 1.5*np.pi, n_tgt) - .5*np.pi*i)+1
        return coeff

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[PatternEncoderBase]:
        pass

    def _get_node(self, src: bool, idx: int) -> Node:
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def get_problem_name(self):
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

    def get_problem_name(self):
        return 'An Comb Prob'

    def __str__(self):
        return f'{self.get_problem_name()} {self._n_src} -> {self._n_tgt}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[PatternEncoderBase]:
        return CombiningPatternEncoder(imputer)


class AnalyticalAssignmentProblem(AnalyticalProblemBase):
    """Assignment pattern: connect any source to any target, optionally with injective/surjective/bijective constraints:
    sources and target have any number of connections, no repetitions;
    - injective: targets have max 1 connection
    - surjective: targets have min 1 connection (note: this is the same as a covering partitioning problem!)
    - bijective (= injective & surjective): targets have exactly 1 connection (same as non-covering partitioning)"""

    def __init__(self, *args, injective=False, surjective=False, repeatable=False, **kwargs):
        self.injective = injective
        self.surjective = surjective
        self.repeatable = repeatable
        super().__init__(*args, **kwargs)
        if (injective and surjective) or repeatable:
            self._invert_f2 = True
        if repeatable:
            self._skew_f2 = True

    def get_init_kwargs(self) -> dict:
        return {**super().get_init_kwargs(), **{
            'injective': self.injective, 'surjective': self.surjective, 'repeatable': self.repeatable}}

    def _get_node(self, src: bool, idx: int) -> Node:
        if src:
            return Node(min_conn=0, repeated_allowed=self.repeatable)
        if self.injective:
            if self.surjective:
                return Node([1], repeated_allowed=self.repeatable)
            return Node([0, 1], repeated_allowed=self.repeatable)
        return Node(min_conn=1 if self.surjective else 0, repeated_allowed=self.repeatable)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_src={self._n_src}, n_tgt={self._n_tgt}, ' \
               f'injective={self.injective}, surjective={self.surjective}, repeatable={self.repeatable})'

    def get_problem_name(self):
        return f'An Assign Prob{"; inj" if self.injective else ""}{"; sur" if self.surjective else ""}' \
               f'{"; rep" if self.repeatable else ""}'

    def __str__(self):
        return f'An Assign Prob {self._n_src} -> {self._n_tgt}{"; inj" if self.injective else ""}' \
               f'{"; sur" if self.surjective else ""}{"; rep" if self.repeatable else ""}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[PatternEncoderBase]:
        if self.injective and self.surjective and not self.repeatable:
            return PartitioningPatternEncoder(imputer)
        if self.injective:
            return DownselectingPatternEncoder(imputer)
        return AssigningPatternEncoder(imputer)


class AnalyticalPartitioningProblem(AnalyticalProblemBase):
    """Partitioning pattern: sources have any connections, targets 1, no repetitions; if changed into a covering
    partitioning pattern, targets have no max connections
    Note: in reverse (and non-covering), this represents multiple combination choices"""

    def __init__(self, *args, covering=False, **kwargs):
        self.covering = covering
        super().__init__(*args, **kwargs)
        if not covering:
            self._invert_f2 = True

    def get_init_kwargs(self) -> dict:
        return {**super().get_init_kwargs(), **{'covering': self.covering}}

    def _get_node(self, src: bool, idx: int) -> Node:
        if src:
            return Node(min_conn=0, repeated_allowed=False)
        return Node(min_conn=1, repeated_allowed=False) if self.covering else Node([1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_src={self._n_src}, n_tgt={self._n_tgt}, ' \
               f'covering={self.covering})'

    def get_problem_name(self):
        return f'An Part Prob{"; cov" if self.covering else ""}'

    def __str__(self):
        return f'An Part Prob {self._n_src} -> {self._n_tgt}{"; cov" if self.covering else ""}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[PatternEncoderBase]:
        if self.covering:
            return AssigningPatternEncoder(imputer)
        return PartitioningPatternEncoder(imputer)


class AnalyticalDownselectingProblem(AnalyticalProblemBase):
    """Downselecting pattern: 1 source selecting one or more of the targets:
    source has any number connection, targets 0 or 1, no repetitions"""

    def __init__(self, encoder=None, n_tgt: int = 3):
        super().__init__(encoder, n_src=1, n_tgt=n_tgt)

    def get_init_kwargs(self) -> dict:
        return {'n_tgt': self._n_tgt}

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node(min_conn=0, repeated_allowed=False) if src else Node([0, 1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_tgt={self._n_tgt})'

    def get_problem_name(self):
        return 'An Down Prob'

    def __str__(self):
        return f'An Down Prob {self._n_src} -> {self._n_tgt}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[PatternEncoderBase]:
        return DownselectingPatternEncoder(imputer)


class AnalyticalConnectingProblem(AnalyticalProblemBase):
    """Connecting pattern: connecting multiple nodes to each other (sources are the same as targets:
    sources and targets have any nr of connections, but cannot be connected to themselves, no repetitions;
    optionally undirected --> only one triangle of the matrix can be connected to"""

    def __init__(self, encoder=None, n: int = 3, directed=True):
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

    def get_problem_name(self):
        return f'An Conn Prob{"; dir" if self.directed else ""}'

    def __str__(self):
        return f'An Conn Prob {self._n_src} -> {self._n_tgt}{"; dir" if self.directed else ""}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[PatternEncoderBase]:
        return ConnectingPatternEncoder(imputer)


class AnalyticalPermutingProblem(AnalyticalProblemBase):
    """Permutation pattern: sources and targets (same amount) have 1 connection each, no repetitions"""

    def __init__(self, encoder=None, n: int = 3):
        super().__init__(encoder, n_src=n, n_tgt=n)

    def get_init_kwargs(self) -> dict:
        return {'n': self._n_src}

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node([1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n={self._n_src})'

    def get_problem_name(self):
        return 'An Perm Prob'

    def __str__(self):
        return f'An Perm Prob {self._n_src} -> {self._n_tgt}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[PatternEncoderBase]:
        return PermutingPatternEncoder(imputer)


class AnalyticalUnorderedNonReplaceCombiningProblem(AnalyticalProblemBase):
    """Unordered non-replacing combining pattern (itertools combinations function: select n_take elements from n_tgt targets):
    1 source has n_take connections to n_tgt targets, no repetition"""

    def __init__(self, encoder=None, n_take: int = 2, n_tgt: int = 3):
        self._n_take = min(n_take, n_tgt)
        super().__init__(encoder, n_src=1, n_tgt=n_tgt)

    def get_init_kwargs(self) -> dict:
        return {'n_take': self._n_take, 'n_tgt': self._n_tgt}

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node([self._n_take], repeated_allowed=False) if src else Node([0, 1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_take={self._n_take}, n_tgt={self._n_tgt})'

    def get_problem_name(self):
        return 'An Unord Norepl Comb Prob'

    def __str__(self):
        return f'An Unord Norepl Comb Prob {self._n_take} from {self._n_tgt}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[PatternEncoderBase]:
        return UnorderedNonReplacingCombiningPatternEncoder(imputer)


class AnalyticalUnorderedCombiningProblem(AnalyticalUnorderedNonReplaceCombiningProblem):
    """Unordered combining (with replacements) pattern (itertools combinations_with_replacement function:
    select n_take elements from n_tgt targets):
    1 source has n_take connections to n_tgt targets, repetition allowed"""

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node([self._n_take]) if src else Node(min_conn=0)

    def get_problem_name(self):
        return 'An Unord Comb Prob'

    def __str__(self):
        return f'An Unord Comb Prob {self._n_take} from {self._n_tgt}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[PatternEncoderBase]:
        return UnorderedCombiningPatternEncoder(imputer)


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

    from assign_enc.selector import *
    from assign_enc.encoder_registry import *
    from assign_pymoo.metrics_compare import *

    # def _do_time():
    #     EncoderSelector.encoding_timeout = .5
    #     EncoderSelector.n_mat_max_eager = 1e3
    #     def _do_cache(disable_cache):
    #         EncoderSelector._global_disable_cache = disable_cache
    #         s = timeit.default_timer()
    #         # enc = AmountFirstGroupedEncoder(DEFAULT_EAGER_IMPUTER(), SourceAmountFlattenedGrouper(), CoordIndexLocationGrouper())
    #         # p = AnalyticalIterCombinationsProblem(enc, n_take=8, n_tgt=16)
    #         p = AnalyticalIterCombinationsProblem(None, n_take=9, n_tgt=19)
    #         print(f'{p!s}: {p.assignment_manager.encoder!s} ({timeit.default_timer()-s:.1f} sec, '
    #               f'imp_ratio = {p.get_imputation_ratio():.2f}, inf_idx = {p.get_information_index():.2f})')
    #     _do_cache(True)
    #     _do_cache(False)
    # _do_time(), exit()

    # p = AnalyticalCombinationProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalAssignmentProblem(EnumRecursiveEncoder(DEFAULT_LAZY_IMPUTER(), n_divide=2), n_src=3, n_tgt=4)
    # p = AnalyticalAssignmentProblem(DEFAULT_LAZY_ENCODER(), n_src=5, n_tgt=5, injective=True)
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER(), n_src=3, n_tg=4)
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER(), injective=True, n_src=2, n_tgt=4)
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER(), injective=True)
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER(), injective=True, n_src=5, n_tgt=5)
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER(), surjective=True)
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER(), injective=True, surjective=True)
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER(), n_src=2, n_tgt=4, repeatable=True)
    # p = AnalyticalPartitioningProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalPartitioningProblem(DEFAULT_EAGER_ENCODER(), covering=True, n_src=3, n_tgt=4)
    # p = AnalyticalPartitioningProblem(LazyAmountFirstEncoder(DEFAULT_LAZY_IMPUTER(), FlatLazyAmountEncoder(), FlatLazyConnectionEncoder()), n_src=2, n_tgt=4, covering=True)
    # p = AnalyticalDownselectingProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalConnectingProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalPermutingProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalUnorderedNonReplaceCombiningProblem(DEFAULT_EAGER_ENCODER(), n_take=5, n_tgt=9)
    p = AnalyticalUnorderedCombiningProblem(DEFAULT_EAGER_ENCODER(), n_take=3, n_tgt=5)

    try:
        # p = p.get_for_encoder(p.get_manual_best_encoder(DEFAULT_LAZY_IMPUTER()))
        p = p.get_for_encoder(ConnIdxGroupedEncoder(DEFAULT_EAGER_IMPUTER()))
        # print([dv.n_opts for dv in p.assignment_manager.encoder.design_vars])
        # from assign_pymoo.sampling import RepairedExhaustiveSampling
        # print(RepairedExhaustiveSampling(p.get_repair()).do(p, 0).get('X'))

    except DetectedHighImpRatio:
        exit()
    print(p.get_imputation_ratio())
    print(p.get_information_index())
    print(p.assignment_manager.encoder.get_distance_correlation())
    exit()

    p.reset_pf_cache()
    p.plot_pf(show_approx_f_range=True, n_sample=1000), exit()
    enc = []
    enc += [e(DEFAULT_EAGER_IMPUTER()) for e in EAGER_ENCODERS]
    enc += [e(DEFAULT_EAGER_IMPUTER()) for e in EAGER_ENUM_ENCODERS]
    enc += [e(DEFAULT_LAZY_IMPUTER()) for e in LAZY_ENCODERS]
    # MetricsComparer().compare_encoders(p, enc)
    MetricsComparer().compare_encoders(p, enc, inf_idx=True)
    # MetricsComparer(n_samples=50, n_leave_out=30).check_information_corr(p, enc)
