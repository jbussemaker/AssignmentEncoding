import numpy as np
from typing import *
from assign_enc.lazy_encoding import *

__all__ = ['EnumOrdinalEncoder']


class EnumOrdinalEncoder(QuasiLazyEncoder):
    """Defines one design variable with the same number of options as the number of possible assignment patterns."""

    def _encode_matrix(self, matrix: np.ndarray) -> List[DiscreteDV]:
        n_mat = matrix.shape[0]
        if n_mat <= 1:
            return []
        return [DiscreteDV(n_opts=n_mat)]

    def _decode_matrix(self, vector: DesignVector, matrix: np.ndarray) -> Optional[Tuple[DesignVector, np.ndarray]]:
        i_mat = vector[0] if len(vector) > 0 else 0
        if i_mat >= matrix.shape[0]:
            return [matrix.shape[0]-1], matrix[-1, :, :]
        return vector, matrix[i_mat, :, :]

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r})'

    def __str__(self):
        return f'Enum Ord + {self._imputer!s}'
