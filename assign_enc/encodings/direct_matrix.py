import numpy as np
from assign_enc.encoding import *

__all__ = ['DirectMatrixEncoder']


class DirectMatrixEncoder(Encoder):
    """Defines one design variable for each matrix element."""

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat, n_src, n_tgt = matrix.shape
        n_dv = n_src*n_tgt

        # Map matrix elements to design vector values
        design_vectors = matrix.reshape(n_mat, n_dv)

        # Move design vector values so that the first value is always 0
        design_vectors -= np.min(design_vectors, axis=0)
        return design_vectors
