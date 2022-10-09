import numpy as np
from typing import *
from assign_enc.encoding import *

__all__ = ['ConstraintViolationImputer']


class ConstraintViolationImputer(EagerImputer):
    """Does not do imputation, but sends a matrix with -1 as values to indicate an invalid design point.
    The optimization problem should represent this as a violated constraint."""

    def impute(self, vector: DesignVector, matrix_mask: MatrixSelectMask) -> Tuple[DesignVector, np.ndarray]:
        invalid_matrix = self._matrix[0, :, :]*0-1
        return vector, invalid_matrix
