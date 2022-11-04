import itertools
import numpy as np
from typing import *
from assign_enc.lazy_encoding import *

__all__ = ['LazyDeltaImputer']


class LazyDeltaImputer(LazyImputer):
    """Imputes by looking for the first valid design vector created by iterating over delta values."""

    def _impute(self, vector: DesignVector, matrix: Optional[np.ndarray], existence: NodeExistence,
                validate: Callable[[np.ndarray], bool], tried_vectors: Set[Tuple[int, ...]]) -> Tuple[DesignVector, np.ndarray]:

        # Determine delta values to try out
        def _sort_by_dist(des_var_delta):
            i_sorted = np.argsort(np.abs(des_var_delta))
            return des_var_delta[i_sorted]

        delta_values = [_sort_by_dist(np.arange(dv.n_opts)-vector[i]) for i, dv in enumerate(self._des_vars)]

        # Loop over all delta values
        first = True
        for dv_delta in itertools.product(*delta_values):
            if first:  # Skip first as this one has a delta of 0
                first = False
                continue
            dv = vector+np.array(dv_delta)

            # Validate associated matrix
            matrix = self._decode(dv, existence)
            if validate(matrix):
                return dv, matrix

        raise RuntimeError('No valid design vector found!')

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'Delta Imp'
