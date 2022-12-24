from assign_enc.matrix import *
from assign_experiments.problems.analytical import *

__all__ = ['AnalyticalCombinationDownselectingProblem', 'AnalyticalPartitioningPermutingProblem',
           'AnalyticalIterCombBothProblem']


class AnalyticalCombinationDownselectingProblem(AnalyticalProblemBase):
    """Combination of the combination and downselecting analytical problems"""

    def __init__(self, encoder, n_tgt: int = 3):
        super().__init__(encoder, n_src=2, n_tgt=n_tgt)

    def get_init_kwargs(self) -> dict:
        return {'n_tgt': self._n_tgt}

    def _get_node(self, src: bool, idx: int) -> Node:
        if src:
            if idx == 0:  # Combination problem
                return Node([1], repeated_allowed=False)
            return Node(min_conn=0, repeated_allowed=False)  # Downselecting problem
        return Node([0, 1, 2], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_tgt={self._n_tgt})'

    def get_problem_name(self):
        return 'An Comb Down Prob'

    def __str__(self):
        return f'{self.get_problem_name()} {self._n_src} -> {self._n_tgt}'


class AnalyticalPartitioningPermutingProblem(AnalyticalProblemBase):
    """Combination of the partitioning and non-covering permuting problems"""

    _invert_f2 = True

    def __init__(self, encoder, n_src: int = 2, n_tgt: int = 3):
        self._n_part_src = n_src
        super().__init__(encoder, n_src=n_src+n_tgt, n_tgt=n_tgt)

    def get_init_kwargs(self) -> dict:
        return {'n_src': self._n_part_src, 'n_tgt': self._n_tgt}

    def _get_node(self, src: bool, idx: int) -> Node:
        if src:
            if idx < self._n_part_src:
                return Node(min_conn=0, repeated_allowed=False)
            return Node([1], repeated_allowed=False)
        return Node([2], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_src={self._n_part_src}, n_tgt={self._n_tgt})'

    def get_problem_name(self):
        return 'An Part Perm Prob'

    def __str__(self):
        return f'{self.get_problem_name()} {self._n_src} -> {self._n_tgt}'


class AnalyticalIterCombBothProblem(AnalyticalProblemBase):
    """Combination of the itertools combinations function, with and without replacement"""

    _invert_f2 = True

    def __init__(self, encoder, n_take: int = 2, n_tgt: int = 3):
        self._n_take = min(n_take, n_tgt)
        super().__init__(encoder, n_src=2, n_tgt=n_tgt)

    def get_init_kwargs(self) -> dict:
        return {'n_take': self._n_take, 'n_tgt': self._n_tgt}

    def _get_node(self, src: bool, idx: int) -> Node:
        if src:
            return Node([self._n_take], repeated_allowed=idx == 0)
        return Node(min_conn=0)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_take={self._n_take}, n_tgt={self._n_tgt})'

    def get_problem_name(self):
        return 'An Iter Comb Both Prob'

    def __str__(self):
        return f'{self.get_problem_name()} {self._n_take} from {self._n_tgt}'


if __name__ == '__main__':
    from assign_enc.encoder_registry import *

    # p = AnalyticalCombinationDownselectingProblem(DEFAULT_EAGER_ENCODER())
    p = AnalyticalPartitioningPermutingProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalIterCombBothProblem(DEFAULT_EAGER_ENCODER())

    p.reset_pf_cache(), p.plot_pf(show_approx_f_range=True), exit()
