import numpy as np
from typing import *
from assign_enc.lazy_encoding import *

__all__ = ['LazyConstraintViolationImputer']


class LazyConstraintViolationImputer(LazyImputer):
    """Does not do imputation, but sends a matrix with -1 as values to indicate an invalid design point.
    The optimization problem should represent this as a violated constraint."""

    def _impute(self, vector: DesignVector, matrix: Optional[np.ndarray], src_exists: np.ndarray,
                tgt_exists: np.ndarray, validate: Callable[[np.ndarray], bool], tried_vectors: Set[Tuple[int, ...]]) \
            -> Tuple[DesignVector, np.ndarray]:
        invalid_matrix = -np.ones((len(src_exists), len(tgt_exists)), dtype=int)
        return vector, invalid_matrix

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'Constr Vio Imp'
