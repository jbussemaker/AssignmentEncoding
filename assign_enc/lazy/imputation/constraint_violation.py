import numpy as np
from typing import *
from assign_enc.lazy_encoding import *

__all__ = ['LazyConstraintViolationImputer']


class LazyConstraintViolationImputer(LazyImputer):
    """Does not do imputation, but sends a matrix with -1 as values to indicate an invalid design point.
    The optimization problem should represent this as a violated constraint."""

    def _impute(self, vector: DesignVector, matrix: np.ndarray, src_exists: np.ndarray, tgt_exists: np.ndarray,
                validate: Callable[[np.ndarray], bool], tried_vectors: Set[Tuple[int, ...]]) -> Tuple[DesignVector, np.ndarray]:
        invalid_matrix = matrix*0-1
        return vector, invalid_matrix
