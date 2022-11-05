import numpy as np
from assign_enc.encoding import *

__all__ = ['OneVarEncoder']


class OneVarEncoder(EagerEncoder):
    """Defines one design variable with the same number of options as the number of possible assignment patterns."""

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        if n_mat <= 1:
            return np.empty((0, 0), dtype=int)
        return np.array([np.arange(0, n_mat)]).T

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r})'

    def __str__(self):
        return f'One Var + {self._imputer!s}'
