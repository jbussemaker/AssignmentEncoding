import numpy as np
from typing import *
from dataclasses import dataclass

__all__ = ['DiscreteDV', 'DesignVector', 'PartialDesignVector', 'MatrixSelectMask', 'Imputer', 'Encoder',
           'filter_design_vectors']


@dataclass
class DiscreteDV:
    n_opts: int


DesignVector = List[int]
PartialDesignVector = List[Optional[int]]
MatrixSelectMask = np.ndarray


def filter_design_vectors(design_vectors: np.ndarray, vector: PartialDesignVector) -> MatrixSelectMask:
    """Filter matrices along the first dimension given a design vector. Returns a mask of selected matrices."""

    dv_mask = np.ones(design_vectors.shape, dtype=bool)
    for i, value in enumerate(vector):
        if value is not None:
            # Select design vectors that have the targeted value for this design variable
            dv_mask[:, i] = design_vectors[:, i] == value

    return np.all(dv_mask, axis=1)


class Imputer:
    """Base class for imputing design vectors to select existing matrices."""

    def __init__(self):
        self._matrix = None
        self._design_vectors = None
        self._design_vars = None

    def initialize(self, matrix: np.ndarray, design_vectors: np.ndarray, design_vars: List[DiscreteDV]):
        self._matrix = matrix
        self._design_vectors = design_vectors
        self._design_vars = design_vars

    def _filter_design_vectors(self, vector: PartialDesignVector) -> MatrixSelectMask:
        return filter_design_vectors(self._design_vectors, vector)

    def impute(self, vector: DesignVector, matrix_mask: MatrixSelectMask) -> Tuple[DesignVector, np.ndarray]:
        """Return a new design vector and associated assignment matrix (n_src x n_tgt)"""
        raise NotImplementedError


class Encoder:
    """Base class that encodes assignment matrices to discrete design variables."""

    def __init__(self, matrix: np.ndarray, imputer: Imputer):
        self._matrix = matrix
        self._n_mat = matrix.shape[0]
        self._design_vectors = self._encode(matrix)
        self._design_vars = self._get_design_variables(self._design_vectors)

        self._imputer = imputer
        imputer.initialize(matrix, self._design_vectors, self._design_vars)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def n_mat(self) -> int:
        return self._n_mat

    @property
    def design_vars(self) -> List[DiscreteDV]:
        return self._design_vars

    def get_matrix(self, vector: DesignVector, matrix_mask: MatrixSelectMask = None) -> Tuple[DesignVector, np.ndarray]:
        """Select a connection matrix (n_src x n_tgt) and impute the design vector if needed."""
        if matrix_mask is None:
            matrix_mask = np.ones((self._matrix.shape[0],), dtype=bool)
        i_mat, = np.where(matrix_mask & self._filter_design_variables(vector))
        if len(i_mat) > 1:
            raise RuntimeError(f'Design vector maps to more than one matrix: {vector}')

        # If no matrix can be found, impute
        if len(i_mat) == 0:
            return self._imputer.impute(vector, matrix_mask)

        # Design vector directly maps to possible matrix
        return vector, self._matrix[i_mat[0], :, :]

    def _filter_design_variables(self, vector: PartialDesignVector) -> MatrixSelectMask:
        return filter_design_vectors(self._design_vectors, vector)

    def _get_design_variables(self, design_vectors: np.ndarray) -> List[DiscreteDV]:
        """Convert possible design vectors to design variable definitions"""

        # Check if all design vectors are unique
        if np.unique(design_vectors, axis=0).shape[0] < design_vectors.shape[0]:
            raise RuntimeError('Not all design vectors are unique!')

        return [DiscreteDV(n_opts=np.max(design_vectors[:, i])+1) for i in range(design_vectors.shape[1])]

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        """
        Encode a matrix of size n_patterns x n_src x n_tgt as discrete design variables.
        Returns the list of design vectors for each matrix in a n_patterns x n_dv array.
        """
        raise NotImplementedError
