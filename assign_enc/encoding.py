import numba
import numpy as np
from typing import *
from dataclasses import dataclass

__all__ = ['DiscreteDV', 'DesignVector', 'PartialDesignVector', 'MatrixSelectMask', 'EagerImputer', 'EagerEncoder',
           'filter_design_vectors', 'flatten_matrix']


@dataclass
class DiscreteDV:
    n_opts: int

    def get_random(self):
        return np.random.randint(0, self.n_opts)


DesignVector = Union[List[int], np.ndarray]
PartialDesignVector = List[Optional[int]]
MatrixSelectMask = np.ndarray


def filter_design_vectors(design_vectors: np.ndarray, vector: PartialDesignVector) -> MatrixSelectMask:
    int_vector = np.array([-1 if val is None else val for val in vector], dtype=int)
    return _filter_design_vectors(design_vectors, int_vector)


@numba.jit(nopython=True, cache=True)
def _filter_design_vectors(design_vectors: np.ndarray, vector: np.ndarray) -> MatrixSelectMask:
    """Filter matrices along the first dimension given a design vector. Returns a mask of selected matrices."""

    dv_mask = np.ones(design_vectors.shape, dtype=numba.types.bool_)
    for i, value in enumerate(vector):
        if value != -1:
            # Select design vectors that have the targeted value for this design variable
            dv_mask[:, i] = design_vectors[:, i] == value

    matrix_mask = np.ones((design_vectors.shape[0],), dtype=numba.types.bool_)
    for i in range(design_vectors.shape[0]):
        matrix_mask[i] = np.all(dv_mask[i, :])
    return matrix_mask


class EagerImputer:
    """Base class for imputing design vectors to select existing matrices."""

    def __init__(self):
        self._matrix: Optional[np.ndarray] = None
        self._design_vectors: Optional[np.ndarray] = None
        self._design_vars: Optional[List[DiscreteDV]] = None

    def initialize(self, matrix: np.ndarray, design_vectors: np.ndarray, design_vars: List[DiscreteDV]):
        self._matrix = matrix
        self._design_vectors = design_vectors
        self._design_vars = design_vars

    def _filter_design_vectors(self, vector: PartialDesignVector) -> MatrixSelectMask:
        return filter_design_vectors(self._design_vectors, vector)

    def _return_imputation(self, i_dv: int) -> Tuple[DesignVector, np.ndarray]:
        return self._design_vectors[i_dv, :], self._matrix[i_dv, :, :]

    def impute(self, vector: DesignVector, matrix_mask: MatrixSelectMask) -> Tuple[DesignVector, np.ndarray]:
        """Return a new design vector and associated assignment matrix (n_src x n_tgt)"""
        raise NotImplementedError


class EagerEncoder:
    """Base class that encodes assignment matrices to discrete design variables."""

    def __init__(self, imputer: EagerImputer, matrix: np.ndarray = None):
        self._matrix = matrix
        self._n_mat = 0
        self._design_vectors = np.array([])
        self._design_vars = []
        self._imputer = imputer

        if matrix is not None:
            self.matrix = matrix

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: np.ndarray):
        self._matrix = matrix
        self._n_mat = matrix.shape[0]
        self._design_vectors = self._encode(matrix)
        self._design_vars = self._get_design_variables(self._design_vectors)
        self._imputer.initialize(matrix, self._design_vectors, self._design_vars)

    @property
    def n_mat(self) -> int:
        return self._n_mat

    @property
    def design_vars(self) -> List[DiscreteDV]:
        return self._design_vars

    def get_random_design_vector(self) -> DesignVector:
        return [dv.get_random() for dv in self.design_vars]

    def get_matrix(self, vector: DesignVector, matrix_mask: MatrixSelectMask = None) -> Tuple[DesignVector, np.ndarray]:
        """Select a connection matrix (n_src x n_tgt) and impute the design vector if needed."""
        i_mat = self.get_matrix_index(vector, matrix_mask=matrix_mask)

        # If no matrix can be found, impute
        if i_mat is None:
            if matrix_mask is None:
                matrix_mask = np.ones((self._matrix.shape[0],), dtype=bool)

            # If the mask filters out all design vectors, there is no need to try imputing
            elif np.all(~matrix_mask):
                null_matrix = self._matrix[0, :, :]*0
                return [0]*len(vector), null_matrix

            return self._imputer.impute(vector, matrix_mask)

        # Design vector directly maps to possible matrix
        return vector, self._matrix[i_mat, :, :]

    def is_valid_vector(self, vector: DesignVector, matrix_mask: MatrixSelectMask = None) -> bool:
        return self.get_matrix_index(vector, matrix_mask=matrix_mask) is not None

    def get_matrix_index(self, vector: DesignVector, matrix_mask: MatrixSelectMask = None) -> Optional[int]:
        if self._matrix is None:
            raise RuntimeError('Matrix not set!')
        if matrix_mask is None:
            matrix_mask = np.ones((self._matrix.shape[0],), dtype=bool)
        i_mat, = np.where(matrix_mask & self._filter_design_variables(vector))
        if len(i_mat) > 1:
            raise RuntimeError(f'Design vector maps to more than one matrix: {vector}')

        # Only if we find exactly one corresponding matrix, it is indeed a valid design vector
        return i_mat[0] if len(i_mat) == 1 else None

    def _filter_design_variables(self, vector: PartialDesignVector) -> MatrixSelectMask:
        return filter_design_vectors(self._design_vectors, vector)

    def _get_design_variables(self, design_vectors: np.ndarray) -> List[DiscreteDV]:
        """Convert possible design vectors to design variable definitions"""

        # Check if all design vectors are unique
        if np.unique(design_vectors, axis=0).shape[0] < design_vectors.shape[0]:
            raise RuntimeError('Not all design vectors are unique!')

        # Check bounds
        if np.min(design_vectors) != 0:
            raise RuntimeError('Design variables should start at zero!')

        return [DiscreteDV(n_opts=np.max(design_vectors[:, i])+1) for i in range(design_vectors.shape[1])]

    @staticmethod
    def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
        return flatten_matrix(matrix)

    @staticmethod
    def _normalize_design_vectors(design_vectors: np.ndarray, remove_gaps=True) -> np.ndarray:
        """Move lowest values to 0 and eliminate value gaps."""

        # Move to zero
        if not remove_gaps:
            return design_vectors-np.min(design_vectors, axis=0)

        # Remove gaps
        design_vectors = design_vectors.copy()
        for i_dv in range(design_vectors.shape[1]):
            des_var = design_vectors[:, i_dv].copy()
            unique_values = np.sort(np.unique(des_var))
            for i_unique, value in enumerate(unique_values):
                design_vectors[des_var == value, i_dv] = i_unique
        return design_vectors

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        """
        Encode a matrix of size n_patterns x n_src x n_tgt as discrete design variables.
        Returns the list of design vectors for each matrix in a n_patterns x n_dv array.
        Assumes values range between 0 and the number of options per design variable.
        """
        raise NotImplementedError


def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
    """Helper function that flattens matrices in the higher dimensions: n_mat x n_src x n_tgt --> n_mat x n_src*n_tgt"""
    return matrix.reshape(matrix.shape[0], np.prod(matrix.shape[1:]))
