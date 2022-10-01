import numpy as np
from typing import *
from assign_enc.encoding import *

__all__ = ['ClosestImputer']


class ClosestImputer(Imputer):
    """Imputes by looking for the closest valid design vector based on euclidean or Mahattan distance."""

    def __init__(self, euclidean=True):
        super().__init__()
        self.euclidean = euclidean

    def impute(self, vector: DesignVector, matrix_mask: MatrixSelectMask) -> Tuple[DesignVector, np.ndarray]:
        design_vectors = self._design_vectors[matrix_mask, :]
        matrices = self._matrix[matrix_mask, :, :]

        elements, target = design_vectors, np.array(vector)
        if self.euclidean:
            dist = self._calc_dist_euclidean(elements, target)
        else:
            dist = self._calc_dist_manhattan(elements, target)
        i_min_dist = np.argmin(dist)

        return design_vectors[i_min_dist, :], matrices[i_min_dist, :, :]

    @staticmethod
    def _calc_dist_euclidean(elements: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((elements-target)**2, axis=1))

    @staticmethod
    def _calc_dist_manhattan(elements: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(elements-target), axis=1)
