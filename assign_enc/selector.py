import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import *
from assign_enc.matrix import *
from assign_enc.time_limiter import *
from assign_enc.lazy_encoding import *
from assign_enc.encoder_registry import *
from assign_enc.assignment_manager import *

__all__ = ['EncoderSelector']

log = logging.getLogger('assign_enc.selector')


class EncoderSelector:
    """Algorithm for automatically selecting the best encoder for a given assignment problem."""

    def __init__(self, src: List[Node], tgt: List[Node], excluded: List[Tuple[Node, Node]] = None,
                 existence_patterns: NodeExistencePatterns = None):
        self.src: List[Node] = src
        self.tgt: List[Node] = tgt
        self._ex: List[Tuple[Node, Node]] = excluded
        self.existence_patterns: Optional[NodeExistencePatterns] = existence_patterns

        self.encoding_timeout = 2  # sec
        self.max_imputation_ratio = 100
        self.min_information_index = .6
        self.n_mat_max_eager = 1000

        self.lazy_imputer = DEFAULT_LAZY_IMPUTER
        self.eager_imputer = DEFAULT_EAGER_IMPUTER

    def reset_cache(self):
        cache_path = self._cache_path(f'{self._get_cache_key()}.pkl')
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def get_best_assignment_manager(self, cache=True) -> AssignmentManager:
        cache_path = self._cache_path(f'{self._get_cache_key()}.pkl')
        if cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as fp:
                return pickle.load(fp)

        assignment_manager = self._get_best_assignment_manager()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as fp:
            pickle.dump(assignment_manager, fp)
        return assignment_manager

    def _get_best_assignment_manager(self) -> AssignmentManager:
        log.info('Counting matrices...')
        n_mat = self._get_n_mat()
        log.info(f'{n_mat} matrices')

        scoring = {'idx': [], 'n_des_pts': [], 'imp_ratio': [], 'inf_idx': []}
        assignment_managers = []

        def _create_managers(encoders, imputer_factory):
            for encoder_factory in encoders:
                # Create encoder
                encoder = encoder_factory(imputer_factory())

                # Create assignment manager
                args = (self.src, self.tgt, encoder)
                kwargs = {'excluded': self._ex, 'existence_patterns': self.existence_patterns}
                cls = LazyAssignmentManager if isinstance(encoder, LazyEncoder) else AssignmentManager

                log.info(f'Encoding {encoder!s}')
                try:
                    with time_limiter(self.encoding_timeout):
                        assignment_manager = cls(*args, **kwargs)
                except (TimeoutError, MemoryError):
                    log.info('Encoding timeout!')
                    continue

                assignment_managers.append(assignment_manager)

                # Get score
                n_design_points = assignment_manager.encoder.get_n_design_points()
                imputation_ratio = (n_design_points/n_mat) if n_mat is not None else n_design_points
                information_index = assignment_manager.encoder.get_information_index()

                scoring['idx'].append(len(scoring['idx']))
                scoring['n_des_pts'].append(n_design_points)
                scoring['imp_ratio'].append(imputation_ratio)
                scoring['inf_idx'].append(information_index)

        # Try lazy encoders
        _create_managers(LAZY_ENCODERS, self.lazy_imputer)
        take_one = False
        if n_mat is not None and n_mat <= self.n_mat_max_eager:
            _create_managers(EAGER_ENCODERS, self.eager_imputer)
            take_one = True
        i_best = self._get_best(pd.DataFrame(data=scoring), knows_n_mat=n_mat is not None, take_one=take_one)
        if i_best is not None:
            return assignment_managers[i_best]

        # Try eager encoders
        _create_managers(EAGER_ENCODERS, self.eager_imputer)
        i_best = self._get_best(pd.DataFrame(data=scoring), knows_n_mat=n_mat is not None, take_one=True)
        return assignment_managers[i_best]

    def _get_best(self, df_scores: pd.DataFrame, knows_n_mat, take_one=False) -> Optional[int]:
        # If we don't accurately know the total amount of matrices, normalize to the lowest imputation ratio
        if not knows_n_mat:
            df_scores.imp_ratio /= df_scores.imp_ratio.min()

        def _return_best_imp_inf(df_):
            # Get rows with the minimum imputation ratio
            min_imp_ratio = np.min(df_.imp_ratio)
            min_imp_ratio_mask = df_.imp_ratio == min_imp_ratio

            # Within these rows, get the row with the maximum information index
            i_max_inf_idx = np.argmax(df_.inf_idx[min_imp_ratio_mask])
            min_imp_idx = df_.idx.values[min_imp_ratio_mask]
            return min_imp_idx[i_max_inf_idx]

        # Filter based on imputation ratio
        df_imp = df_scores[df_scores.imp_ratio == 1.]
        if len(df_imp) == 0:
            df_imp = df_scores[df_scores.imp_ratio <= self.max_imputation_ratio]
        if len(df_imp) == 0:
            return _return_best_imp_inf(df_scores) if take_one else None

        # Filter based on information index
        df_inf_idx = df_imp[df_imp.inf_idx >= self.min_information_index]
        if len(df_inf_idx) == 0:
            return _return_best_imp_inf(df_imp) if take_one else None

        # From the remaining encoders, take the one with the lowest imputation ratio
        return _return_best_imp_inf(df_inf_idx)

    def _get_n_mat(self) -> Optional[int]:
        matrix_gen = AggregateAssignmentMatrixGenerator(
            self.src, self.tgt, excluded=self._ex, existence_patterns=self.existence_patterns)
        try:
            with time_limiter(self.encoding_timeout):
                return matrix_gen.count_all_matrices(max_by_existence=True)
        except TimeoutError:
            pass

    def _get_cache_key(self):
        return AggregateAssignmentMatrixGenerator.get_cache_key(self.src, self.tgt, self._ex, self.existence_patterns)

    def _cache_path(self, sub_path=None):
        cache_folder = os.path.join(os.path.dirname(__file__), '.encoder_cache')
        return cache_folder if sub_path is None else os.path.join(cache_folder, sub_path)
