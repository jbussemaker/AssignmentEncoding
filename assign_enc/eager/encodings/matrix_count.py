import itertools
import numpy as np
from assign_enc.encoding import *

__all__ = ['OneVarEncoder', 'RecursiveEncoder']


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


class RecursiveEncoder(EagerEncoder):
    """Defines design variables where each has a fixed nr of options to cover all possible assignment patterns."""

    def __init__(self, *args, n_divide=2, **kwargs):
        self.n_divide = max(2, n_divide)
        super().__init__(*args, **kwargs)

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        if n_mat <= 1:
            return np.zeros((0, 0), dtype=int)

        n = self.n_divide
        n_var = int(np.ceil(np.log(n_mat)/np.log(n)))
        design_vectors = np.array(list(itertools.product(*[np.arange(n) for _ in range(n_var)]))[:matrix.shape[0]])
        return design_vectors

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r}, n_divide={self.n_divide})'

    def __str__(self):
        return f'Recursive {self.n_divide} + {self._imputer!s}'
