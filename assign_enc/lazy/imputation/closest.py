import itertools
import numpy as np
from typing import *
from assign_enc.lazy_encoding import *

__all__ = ['LazyClosestImputer']


class LazyClosestImputer(LazyImputer):
    """Imputes by looking for the closest valid design vector based on euclidean or Mahattan distance.
    Eagerly evaluates itertools.product, so might need too much memory/time."""

    def __init__(self, euclidean=False):
        self.euclidean = euclidean
        super().__init__()

    def _impute(self, vector: DesignVector, matrix: np.ndarray, src_exists: np.ndarray, tgt_exists: np.ndarray,
                validate: Callable[[np.ndarray], bool], tried_vectors: Set[Tuple[int, ...]]) -> Tuple[DesignVector, np.ndarray]:

        # Determine delta values to try out
        def _sort_by_dist(des_var_delta):
            i_sorted = np.argsort(np.abs(des_var_delta))
            return des_var_delta[i_sorted]

        delta_values = [_sort_by_dist(np.arange(dv.n_opts)-vector[i]) for i, dv in enumerate(self._des_vars)]

        # Sort all delta values by distance (skip first as it has a delta of 0)
        dist_func = self._calc_dist_euclidean if self.euclidean else self._calc_dist_manhattan
        dv_deltas = [np.array(dv_delta) for dv_delta in itertools.product(*delta_values)][1:]  # Skip first as it has a delta of 0
        dv_dist = dist_func(np.array(dv_deltas))

        # Sort by distance
        i_dv_dist = np.argsort(dv_dist)
        for i_dv in i_dv_dist:
            dv = vector+dv_deltas[i_dv]

            # Validate associated matrix
            matrix = self._decode(dv, src_exists, tgt_exists)
            if validate(matrix):
                return dv, matrix

        raise RuntimeError('No valid design vector found!')

    @staticmethod
    def _calc_dist_euclidean(deltas: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(deltas**2, axis=1))

    @staticmethod
    def _calc_dist_manhattan(deltas: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(deltas), axis=1)
