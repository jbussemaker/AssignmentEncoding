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
from assign_enc.cache import get_cache_path

__all__ = ['EncoderSelector']

log = logging.getLogger('assign_enc.selector')


class EncoderSelector:
    """Algorithm for automatically selecting the best encoder for a given assignment problem."""

    _global_disable_cache = False  # For testing/experiments

    encoding_timeout = .15  # sec
    n_mat_max_eager = 1e3
    imputation_ratio_limits = [10, 40, 80, 200]
    min_information_index = .6

    def __init__(self, src: List[Node], tgt: List[Node], excluded: List[Tuple[Node, Node]] = None,
                 existence_patterns: NodeExistencePatterns = None):
        self.src: List[Node] = src
        self.tgt: List[Node] = tgt
        self._ex: List[Tuple[Node, Node]] = excluded
        self.existence_patterns: Optional[NodeExistencePatterns] = existence_patterns

        self.lazy_imputer = DEFAULT_LAZY_IMPUTER
        self.eager_imputer = DEFAULT_EAGER_IMPUTER

    def reset_cache(self):
        cache_path = self._cache_path(f'{self._get_cache_key()}.pkl')
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def get_best_assignment_manager(self, cache=True, limit_time=True) -> AssignmentManager:
        cache_path = self._cache_path(f'{self._get_cache_key()}.pkl')
        if not self._global_disable_cache and cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as fp:
                return pickle.load(fp)

        enc_timeout, n_mme = self.encoding_timeout, self.n_mat_max_eager
        if not limit_time:
            self.encoding_timeout = 5
            self.n_mat_max_eager = 1e5

        assignment_manager = self._get_best_assignment_manager()

        if not limit_time:
            self.encoding_timeout, self.n_mat_max_eager = enc_timeout, n_mme

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as fp:
            pickle.dump(assignment_manager, fp)
        return assignment_manager

    def _get_best_assignment_manager(self) -> AssignmentManager:
        log.info('Counting matrices...')
        self._provision_agg_matrix_cache()
        n_mat, n_exist = self._get_n_mat()
        log.info(f'{n_mat} matrices ({n_exist} existence scheme{"s" if n_exist != 1 else ""})')

        def _instantiate_manager(encoder):
            args = (self.src, self.tgt, encoder)
            kwargs = {'excluded': self._ex, 'existence_patterns': self.existence_patterns}
            cls = LazyAssignmentManager if isinstance(encoder, LazyEncoder) else AssignmentManager

            log.info(f'Encoding {encoder!s}')
            try:
                with time_limiter(self.encoding_timeout):
                    return cls(*args, **kwargs)
            except (TimeoutError, MemoryError):
                log.info('Encoding timeout!')

        def _create_managers(encoders, imputer_factory):
            assignment_mgr = []
            scoring = {'n_des_pts': [], 'imp_ratio': [], 'inf_idx': []}
            for encoder_factory in encoders:
                # Create encoder
                encoder = encoder_factory(imputer_factory())

                # Create assignment manager
                assignment_manager = _instantiate_manager(encoder)
                if assignment_manager is None:
                    continue

                assignment_mgr.append(assignment_manager)

                # Get score
                n_design_points = assignment_manager.encoder.get_n_design_points()
                imputation_ratio = self._get_imp_ratio(n_design_points, n_mat, n_exist, assignment_manager=assignment_manager)
                information_index = assignment_manager.encoder.get_information_index()

                scoring['n_des_pts'].append(n_design_points)
                scoring['imp_ratio'].append(imputation_ratio)
                scoring['inf_idx'].append(information_index)
            return pd.DataFrame(data=scoring), assignment_mgr

        # Special case if there are no possible connections
        if n_mat == 0:
            selected_encoder = DEFAULT_EAGER_ENCODER()
            return _instantiate_manager(selected_encoder)

        # Try lazy encoders
        df_score, assignment_managers = _create_managers(LAZY_ENCODERS, self.lazy_imputer)
        if n_mat is None or n_mat > self.n_mat_max_eager:
            i_best = self._get_best(df_score, knows_n_mat=n_mat is not None, n_priority=4)
            if i_best is not None:
                return assignment_managers[i_best]

        # Try eager encoders
        df_eager, eager_assignment_mgr = _create_managers(EAGER_ENCODERS, self.eager_imputer)
        df_score = pd.concat([df_eager, df_score], ignore_index=True)  # Give preference to eager encoders due to faster imputation
        assignment_managers = eager_assignment_mgr+assignment_managers
        i_best = self._get_best(df_score, knows_n_mat=n_mat is not None, n_priority=8)
        if i_best is not None:
            return assignment_managers[i_best]

        # Try eager enumeration-based encoders
        df_enum, enum_assignment_mgr = _create_managers(EAGER_ENUM_ENCODERS, self.eager_imputer)
        df_score = pd.concat([df_score, df_enum], ignore_index=True)
        assignment_managers = assignment_managers+enum_assignment_mgr
        i_best = self._get_best(df_score, knows_n_mat=n_mat is not None)
        if i_best is None:
            raise RuntimeError(f'Cannot find best encoder, try increasing timeout')
        return assignment_managers[i_best]

    def _get_imp_ratio(self, n_design_points: int, n_mat: int = None, n_exist: int = None,
                       assignment_manager: AssignmentManagerBase = None) -> float:
        if assignment_manager is not None:
            return assignment_manager.encoder.get_imputation_ratio(per_existence=True)
        return ((n_design_points*n_exist)/n_mat) if n_mat is not None else n_design_points

    def reset_agg_matrix_cache(self):
        self._get_matrix_gen().reset_agg_matrix_cache()

    def _provision_agg_matrix_cache(self):
        log.info('Generating aggregate matrix for eager encoders...')
        self._get_matrix_gen().get_agg_matrix(cache=True)

    def _get_best(self, df_scores: pd.DataFrame, knows_n_mat, n_priority: int = None) -> Optional[int]:
        if len(df_scores) == 0:
            return
        df_scores = df_scores.copy()
        df_scores['idx'] = df_scores.index

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
            return int(min_imp_idx[i_max_inf_idx])

        # Check for points with low imputation ratio first
        df_imp_bands = [df_scores[df_scores.imp_ratio == 1]]
        df_imp_bands += [df_scores[df_scores.imp_ratio <= ir_limit] for ir_limit in self.imputation_ratio_limits]
        df_imp_bands += [df_scores]

        df_priority = [
            df_imp_bands[0][df_imp_bands[0].inf_idx >= self.min_information_index],
            df_imp_bands[1][df_imp_bands[1].inf_idx >= self.min_information_index],
            df_imp_bands[0][df_imp_bands[0].inf_idx >= .5*self.min_information_index],
            df_imp_bands[1][df_imp_bands[1].inf_idx >= .5*self.min_information_index],
            df_imp_bands[0][df_imp_bands[0].inf_idx > 0],
            df_imp_bands[1][df_imp_bands[1].inf_idx > 0],
            df_imp_bands[0],
            df_imp_bands[1],
        ]
        for i in range(2, len(df_imp_bands)):
            df_priority += [
                df_imp_bands[i][df_imp_bands[i].inf_idx >= self.min_information_index],
                df_imp_bands[i][df_imp_bands[i].inf_idx >= .5*self.min_information_index],
                df_imp_bands[i][df_imp_bands[i].inf_idx > 0],
                df_imp_bands[i],
            ]

        for i, df in enumerate(df_priority):
            if n_priority is not None and i >= n_priority:
                return
            if len(df) > 0:
                return _return_best_imp_inf(df)
        raise RuntimeError('No encoders available!')

    def _get_n_mat(self) -> Tuple[Optional[int], Optional[int]]:
        try:
            matrix_gen = self._get_matrix_gen()
            with time_limiter(self.encoding_timeout):
                n_mat_total = matrix_gen.count_all_matrices(max_by_existence=False)
            n_existence = len(list(matrix_gen.iter_existence()))
            return n_mat_total, n_existence
        except TimeoutError:
            return None, None

    def _get_matrix_gen(self) -> AggregateAssignmentMatrixGenerator:
        return AggregateAssignmentMatrixGenerator(
            self.src, self.tgt, excluded=self._ex, existence_patterns=self.existence_patterns)

    def _get_cache_key(self):
        ex_idx = AggregateAssignmentMatrixGenerator.ex_to_idx(self.src, self.tgt, self._ex)
        return AggregateAssignmentMatrixGenerator.get_cache_key(self.src, self.tgt, ex_idx, self.existence_patterns)

    def _cache_path(self, sub_path=None):
        sel_cache_folder = 'encoder_cache'
        sub_path = os.path.join(sel_cache_folder, sub_path) if sub_path is not None else sel_cache_folder
        return get_cache_path(sub_path=sub_path)
