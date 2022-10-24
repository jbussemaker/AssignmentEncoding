import numpy as np
from typing import *
from assign_enc.encoding import *

__all__ = ['FirstImputer']


class FirstImputer(EagerImputer):
    """Imputer that simply chooses the first possible matrix."""

    def impute(self, vector: DesignVector, matrix_mask: MatrixSelectMask) -> Tuple[DesignVector, np.ndarray]:
        i_mat = np.where(matrix_mask)[0][0]
        return self._return_imputation(i_mat)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'First Imp'
