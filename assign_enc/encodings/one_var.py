import numpy as np
from assign_enc.encoding import *

__all__ = ['OneVarEncoder']


class OneVarEncoder(Encoder):
    """Defines one design variable with the same number of options as the number of possible assignment patterns."""

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        return np.array([np.arange(0, n_mat)]).T
