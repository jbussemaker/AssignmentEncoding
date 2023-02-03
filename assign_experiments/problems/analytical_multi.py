import itertools
import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_experiments.problems.analytical import *

__all__ = ['MultiAnalyticalProblemBase', 'MultiCombinationProblem', 'MultiAssignmentProblem',
           'MultiPermIterCombProblem']


class MultiAnalyticalProblemBase(AnalyticalProblemBase):

    def __init__(self, encoder, n_src: int = 2, n_tgt: int = 3):
        self._exist_pattern_map, self._aux_dvs, self._aux_des_vars = self._encode_existence_patterns(n_src, n_tgt)
        super().__init__(encoder, n_src=n_src, n_tgt=n_tgt)

    def get_aux_des_vars(self) -> Optional[List[DiscreteDV]]:
        return self._aux_des_vars

    def correct_x_aux(self, x_aux: DesignVector) -> Tuple[DesignVector, Optional[NodeExistence], bool]:
        if tuple(x_aux) in self._exist_pattern_map:
            return x_aux, self._exist_pattern_map[tuple(x_aux)], False

        # Correct the aux vector: use the closest imputation method
        elements, target = self._aux_dvs, np.array(x_aux)
        dist = np.sqrt(np.sum((elements-target)**2, axis=1))
        i_min_dist = np.argmin(dist)
        x_aux_corr = list(self._aux_dvs[i_min_dist, :])
        return x_aux_corr, self._exist_pattern_map[tuple(x_aux_corr)], False

    def get_existence_patterns(self) -> Optional[NodeExistencePatterns]:
        return NodeExistencePatterns(patterns=list(self._exist_pattern_map.values()))

    def _encode_existence_patterns(self, n_src: int, n_tgt: int) \
            -> Tuple[Dict[tuple, NodeExistence], np.ndarray, List[DiscreteDV]]:
        dvs, existence_patterns = zip(*self._create_existence_patterns(n_src, n_tgt))
        aux_des_vector_arr = EagerEncoder.normalize_design_vectors(np.array(dvs, dtype=int))

        exist_pattern_map = {tuple(dv): existence_patterns[i] for i, dv in enumerate(aux_des_vector_arr)}
        if len(exist_pattern_map) != len(existence_patterns):
            raise RuntimeError('Duplicate existence pattern aux vectors!')

        aux_dvs = EagerEncoder.get_design_variables({NodeExistence(): aux_des_vector_arr})
        return exist_pattern_map, aux_des_vector_arr, aux_dvs

    def _create_existence_patterns(self, n_src: int, n_tgt: int) -> List[Tuple[Tuple[int, ...], NodeExistence]]:
        """Note: at the moment of execution, not all class variables have been initialized yet!"""
        raise NotImplementedError

    def get_init_kwargs(self) -> dict:
        raise NotImplementedError

    def _get_node(self, src: bool, idx: int) -> Node:
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def get_problem_name(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class MultiCombinationProblem(MultiAnalyticalProblemBase):
    """Multiple independent combination problems"""

    def __init__(self, encoder, n_multi: int = 3, n_tgt: int = 10):
        super().__init__(encoder, n_src=n_multi, n_tgt=n_tgt)
        self._skew_f2 = True

    def get_init_kwargs(self) -> dict:
        return {'n_multi': self._n_src, 'n_tgt': self._n_tgt}

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node([1], repeated_allowed=False) if src else Node([0, 1], repeated_allowed=False)

    def _create_existence_patterns(self, n_src: int, n_tgt: int) -> List[Tuple[Tuple[int, ...], NodeExistence]]:
        return [((i_exist,), NodeExistence(src_exists=[i == i_exist for i in range(n_src)]))
                for i_exist in range(n_src)]

    def __repr__(self):
        return f'{self.__class__.__name__}(n_multi={self._n_src}, n_tgt={self._n_tgt})'

    def get_problem_name(self):
        return 'Multi Comb Prob'

    def __str__(self):
        return f'{self.get_problem_name()} {self._n_src} -> {self._n_tgt}'


class MultiAssignmentProblem(MultiAnalyticalProblemBase):
    """Assignment problem where parts of the sources and/or targets may be deactivated"""

    def __init__(self, *args, injective=False, surjective=False, repeatable=False, n_act_src: int = 2,
                 n_act_tgt: int = 3, n_src: int = 3, n_tgt: int = 4):
        self.injective = injective
        self.surjective = surjective
        self.repeatable = repeatable
        self._n_act_src = n_act_src
        self._n_act_tgt = n_act_tgt
        super().__init__(*args, n_src=n_src, n_tgt=n_tgt)
        if (injective and surjective) or repeatable:
            self._invert_f2 = True
        if repeatable:
            self._skew_f2 = True

    def get_init_kwargs(self) -> dict:
        return {'injective': self.injective, 'surjective': self.surjective, 'repeatable': self.repeatable,
                'n_act_src': self._n_act_src, 'n_act_tgt': self._n_act_tgt, 'n_src': self._n_src, 'n_tgt': self._n_tgt}

    def _create_existence_patterns(self, n_src: int, n_tgt: int) -> List[Tuple[Tuple[int, ...], NodeExistence]]:
        if self._n_act_src >= n_src:
            src_exists = [None]
        else:
            src_exists = [[i in i_src_act for i in range(n_src)]
                          for i_src_act in itertools.combinations(range(n_src), self._n_act_src)]
        if self._n_act_tgt >= n_tgt:
            tgt_exists = [None]
        else:
            tgt_exists = [[i in i_src_tgt for i in range(n_tgt)]
                          for i_src_tgt in itertools.combinations(range(n_tgt), self._n_act_tgt)]

        patterns = []
        for i_src_exist, i_tgt_exist in itertools.product(range(len(src_exists)), range(len(tgt_exists))):
            pattern = NodeExistence(src_exists=src_exists[i_src_exist], tgt_exists=tgt_exists[i_tgt_exist])
            patterns.append(((i_src_exist, i_tgt_exist), pattern))
        return patterns

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
               f'injective={self.injective}, surjective={self.surjective}, repeatable={self.repeatable}, ' \
               f'n_act_src={self._n_act_src}, n_act_tgt={self._n_act_tgt})'

    def get_problem_name(self):
        return f'Multi Assign Prob{"; inj" if self.injective else ""}{"; sur" if self.surjective else ""}' \
               f'{"; rep" if self.repeatable else ""}'

    def __str__(self):
        return f'Multi Assign Prob {self._n_src} ({self._n_act_src}) -> {self._n_tgt} ({self._n_act_tgt})' \
               f'{"; inj" if self.injective else ""}{"; sur" if self.surjective else ""}{"; rep" if self.repeatable else ""}'


class MultiPermIterCombProblem(MultiAnalyticalProblemBase):
    """Problem that is either a permutation or an itertools combinations problems: these problems have very different
    'optimal' encoders, so they should test the boundaries of the encoder selector."""

    def __init__(self, encoder, n_take: int = 3, n: int = 6):
        self._n_take = n_take
        super().__init__(encoder, n_src=n, n_tgt=n)

    def get_init_kwargs(self) -> dict:
        return {'n_take': self._n_take, 'n': self._n_tgt}

    def _create_existence_patterns(self, n_src: int, n_tgt: int) -> List[Tuple[Tuple[int, ...], NodeExistence]]:
        permutation = NodeExistence(src_n_conn_override={i: [1] for i in range(n_src)},
                                    tgt_n_conn_override={i: [1] for i in range(n_tgt)})
        iter_comb = NodeExistence(src_n_conn_override={i: [self._n_take if i == 0 else 0] for i in range(n_src)})
        return [
            ((0,), permutation),
            ((1,), iter_comb),
        ]

    def _get_node(self, src: bool, idx: int) -> Node:
        if src:
            n_opt = [1] if self._n_take == 1 else [1, self._n_take]
            return Node(n_opt, repeated_allowed=False)
        return Node([0, 1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_take={self._n_take}, n={self._n_tgt})'

    def get_problem_name(self):
        return 'Multi Perm Iter Comb Prob'

    def __str__(self):
        return f'{self.get_problem_name()} {self._n_take} from {self._n_tgt}'


if __name__ == '__main__':
    from assign_enc.encoder_registry import *
    from assign_pymoo.metrics_compare import *

    # p = MultiCombinationProblem(DEFAULT_EAGER_ENCODER())
    # p = MultiAssignmentProblem(DEFAULT_EAGER_ENCODER())
    p = MultiPermIterCombProblem(DEFAULT_EAGER_ENCODER())

    p.reset_pf_cache(), p.plot_pf(show_approx_f_range=True, n_sample=1000), exit()
    enc = []
    enc += [e(DEFAULT_EAGER_IMPUTER()) for e in EAGER_ENCODERS]
    enc += [e(DEFAULT_EAGER_IMPUTER()) for e in EAGER_ENUM_ENCODERS]
    enc += [e(DEFAULT_LAZY_IMPUTER()) for e in LAZY_ENCODERS]
    MetricsComparer().compare_encoders(p, enc, inf_idx=True)
