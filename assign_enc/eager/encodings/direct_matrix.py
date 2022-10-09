import numpy as np
from assign_enc.encoding import *

__all__ = ['DirectMatrixEncoder']


class DirectMatrixEncoder(EagerEncoder):
    """Defines one design variable for each matrix element."""

    def __init__(self, *args, remove_gaps=True, **kwargs):
        self.remove_gaps = remove_gaps
        super().__init__(*args, **kwargs)

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        # Map matrix elements to design vector values
        design_vectors = self.flatten_matrix(matrix)

        # Normalize design vectors (move to 0 and optionally remove value gaps)
        return self._normalize_design_vectors(design_vectors, remove_gaps=self.remove_gaps)
