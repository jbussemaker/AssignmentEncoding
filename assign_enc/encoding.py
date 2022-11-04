import numba
import numpy as np
from typing import *
from assign_enc.matrix import *
from dataclasses import dataclass

__all__ = ['DiscreteDV', 'DesignVector', 'PartialDesignVector', 'MatrixSelectMask', 'EagerImputer', 'Encoder',
           'EagerEncoder', 'filter_design_vectors', 'flatten_matrix', 'NodeExistence']


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


@numba.njit()
def _filter_design_vectors(design_vectors: np.ndarray, vector: np.ndarray) -> MatrixSelectMask:
    """Filter matrices along the first dimension given a design vector. Returns a mask of selected matrices."""

    dv_mask = np.ones((design_vectors.shape[0],), dtype=numba.types.bool_)
    for i, value in enumerate(vector):
        if value != -1:
            # Select design vectors that have the targeted value for this design variable
            dv_mask[dv_mask] = np.bitwise_and(dv_mask[dv_mask], design_vectors[dv_mask, i] == value)
    return dv_mask


class EagerImputer:
    """Base class for imputing design vectors to select existing matrices."""

    def __init__(self):
        self._matrix: Optional[Dict[NodeExistence, np.ndarray]] = None
        self._design_vectors: Optional[Dict[NodeExistence, np.ndarray]] = None
        self._design_vars: Optional[List[DiscreteDV]] = None

    def initialize(self, matrix: Dict[NodeExistence, np.ndarray], design_vectors: Dict[NodeExistence, np.ndarray],
                   design_vars: List[DiscreteDV]):
        self._matrix = matrix
        self._design_vectors = design_vectors
        self._design_vars = design_vars

    def _get_design_vectors(self, existence: NodeExistence) -> np.ndarray:
        if existence not in self._design_vectors:
            return np.empty((0, 0), dtype=int)
        return self._design_vectors[existence]

    def _get_matrix(self, existence: NodeExistence) -> np.ndarray:
        return EagerEncoder.get_matrix_for_existence(self._matrix, existence)[0]

    def _filter_design_vectors(self, vector: PartialDesignVector, existence: NodeExistence) -> MatrixSelectMask:
        design_vectors = self._get_design_vectors(existence)
        vector = vector[:design_vectors.shape[1]]
        return filter_design_vectors(design_vectors, vector)

    def _return_imputation(self, i_dv: int, existence: NodeExistence) -> Tuple[DesignVector, np.ndarray]:
        design_vectors = self._get_design_vectors(existence)
        matrix = self._get_matrix(existence)
        return design_vectors[i_dv, :], matrix[i_dv, :, :]

    def impute(self, vector: DesignVector, existence: NodeExistence, matrix_mask: MatrixSelectMask) \
            -> Tuple[DesignVector, np.ndarray]:
        """Return a new design vector and associated assignment matrix (n_src x n_tgt)"""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class Encoder:

    @property
    def design_vars(self) -> List[DiscreteDV]:
        raise NotImplementedError

    def get_random_design_vector(self) -> DesignVector:
        return [dv.get_random() for dv in self.design_vars]

    def get_n_design_points(self) -> int:
        return int(np.cumprod([dv.n_opts for dv in self.design_vars])[-1])

    def get_imputation_ratio(self) -> float:
        """Ratio of the total design space size to the actual amount of possible connections"""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class EagerEncoder(Encoder):
    """Base class that encodes assignment matrices to discrete design variables."""

    def __init__(self, imputer: EagerImputer, matrix: np.ndarray = None):
        self._matrix = {}
        self._design_vectors = {}
        self._design_vector_map = {}
        self._design_vars = []
        self._imputer = imputer

        if matrix is not None:
            self.matrix = matrix

    @property
    def matrix(self) -> Dict[NodeExistence, np.ndarray]:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: MatrixMapOptional):
        if isinstance(matrix, np.ndarray):
            matrix = {NodeExistence(): matrix}

        # Encode separately for each existence mode
        self._matrix = matrix
        self._design_vectors = des_vectors = {existence: self._encode(mat) for existence, mat in matrix.items()}
        self._design_vector_map = {existence: {tuple(dv): i for i, dv in enumerate(des_vec)}
                                   for existence, des_vec in des_vectors.items()}
        self._design_vars = self._get_design_variables(des_vectors)
        self._imputer.initialize(matrix, self._design_vectors, self._design_vars)

    @property
    def n_mat_max(self) -> int:
        return max([matrix.shape[0] for matrix in self._matrix.values()])

    @property
    def design_vars(self) -> List[DiscreteDV]:
        return self._design_vars

    def get_imputation_ratio(self) -> float:
        n_design_points = self.get_n_design_points()
        n_total = 0
        n_valid = 0
        for matrix in self._matrix.values():
            n_total += n_design_points
            n_valid += matrix.shape[0]
        return n_total/n_valid

    def get_matrix(self, vector: DesignVector, existence: NodeExistence = None,
                   matrix_mask: MatrixSelectMask = None) -> Tuple[DesignVector, np.ndarray]:
        """Select a connection matrix (n_src x n_tgt) and impute the design vector if needed."""

        i_mat, existence = self.get_matrix_index(vector, existence=existence, matrix_mask=matrix_mask)
        matrix, existence = self._get_matrix_for_existence(existence)

        # If this existence mode has no matrices, return the zero vector
        if not self._has_existence(existence):
            vector = [0]*len(vector)
            null_matrix = np.zeros((1, matrix.shape[1], matrix.shape[2]), dtype=int)
            return vector, null_matrix

        # If no matrix can be found, impute
        if i_mat is None:
            if matrix_mask is None:
                matrix_mask = np.ones((matrix.shape[0],), dtype=bool)

            # If the mask filters out all design vectors, there is no need to try imputing
            elif np.all(~matrix_mask):
                null_matrix = matrix[0, :, :]*0
                return [0]*len(vector), null_matrix

            return self._imputer.impute(vector, existence, matrix_mask)

        # Design vector directly maps to possible matrix
        vector = self._correct_vector(vector, existence)
        return vector, matrix[i_mat, :, :]

    def is_valid_vector(self, vector: DesignVector, existence: NodeExistence = None,
                        matrix_mask: MatrixSelectMask = None) -> bool:
        i_mat, existence = self.get_matrix_index(vector, existence=existence, matrix_mask=matrix_mask)
        if i_mat is None:
            return False
        corr_vector = self._correct_vector(vector, existence)
        return np.all(corr_vector == vector)

    def get_matrix_index(self, vector: DesignVector, existence: NodeExistence = None,
                         matrix_mask: MatrixSelectMask = None) -> Tuple[Optional[int], NodeExistence]:
        matrix, existence = self._get_matrix_for_existence(existence)
        if matrix_mask is None:
            matrix_mask = np.ones((matrix.shape[0],), dtype=bool)
        i_mat, = np.where(matrix_mask & self._filter_full_design_vector(matrix, vector, existence=existence))
        if len(i_mat) > 1:
            raise RuntimeError(f'Design vector maps to more than one matrix: {vector}')

        # Only if we find exactly one corresponding matrix, it is indeed a valid design vector
        return i_mat[0] if len(i_mat) == 1 else None, existence

    def _has_existence(self, existence: NodeExistence = None):
        if existence is None:
            existence = NodeExistence()
        return existence in self._matrix

    def _get_matrix_for_existence(self, existence: NodeExistence = None) -> Tuple[np.ndarray, NodeExistence]:
        if len(self._matrix) == 0:
            raise RuntimeError('Matrix not set!')
        return self.get_matrix_for_existence(self._matrix, existence=existence)

    def _correct_vector(self, vector: DesignVector, existence: NodeExistence) -> DesignVector:
        """Set unused design variables to zero (can happen when we are in an existence mode with less design variables
        than the max)"""
        n_dv = 0
        if existence in self._design_vectors:
            n_dv = self._design_vectors[existence].shape[1]

        corrected_vector = vector.copy()
        for i_dv in range(n_dv, len(vector)):
            corrected_vector[i_dv] = 0
        return corrected_vector

    @staticmethod
    def get_matrix_for_existence(matrix_map: MatrixMap, existence: NodeExistence = None)\
            -> Tuple[np.ndarray, NodeExistence]:
        if existence is None:
            existence = NodeExistence()
        if existence not in matrix_map:
            first_matrix = list(matrix_map.values())[0]
            return np.empty((0, first_matrix.shape[1], first_matrix.shape[2]), dtype=int), existence
        return matrix_map[existence], existence

    def _filter_full_design_vector(self, matrix: np.ndarray, vector: DesignVector,
                                   existence: NodeExistence = None) -> MatrixSelectMask:
        if existence is None:
            existence = NodeExistence()

        mask = np.zeros((matrix.shape[0],), dtype=bool)
        if existence not in self._design_vector_map:
            return mask
        design_vectors = self._design_vector_map[existence]
        n_dv = self._design_vectors[existence].shape[1]

        i_mat = None
        if n_dv == 0:
            if len(self._matrix[existence]) == 1:
                i_mat = 0
        else:
            i_mat = design_vectors.get(tuple(vector[:n_dv]))

        if i_mat is not None:
            mask[i_mat] = True
        return mask

    def _get_design_variables(self, design_vectors: Dict[NodeExistence, np.ndarray]) -> List[DiscreteDV]:
        """Convert possible design vectors to design variable definitions"""
        n_max = max([dv.shape[1] for dv in design_vectors.values()])
        n_opts_max = np.zeros((n_max,), dtype=int)

        for des_vectors in design_vectors.values():
            if des_vectors.shape[1] == 0:
                continue

            # Check if all design vectors are unique
            if np.unique(des_vectors, axis=0).shape[0] < des_vectors.shape[0]:
                raise RuntimeError('Not all design vectors are unique!')

            # Check bounds
            if np.min(des_vectors) != 0:
                raise RuntimeError('Design variables should start at zero!')

            n_opts = np.max(des_vectors, axis=0)+1
            n_dv = len(n_opts)
            n_opts_max[:n_dv] = np.max(np.row_stack([n_opts, n_opts_max[:n_dv]]), axis=0)

        # Check number of options
        if min(n_opts_max) <= 1:
            raise RuntimeError('All design variables must have at least two options')

        return [DiscreteDV(n_opts=n) for n in n_opts_max]

    @staticmethod
    def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
        return flatten_matrix(matrix)

    @staticmethod
    def _normalize_design_vectors(design_vectors: np.ndarray, remove_gaps=True) -> np.ndarray:
        """Move lowest values to 0 and eliminate value gaps."""

        # Move to zero
        if not remove_gaps:
            design_vectors -= np.min(design_vectors, axis=0)

        # Remove gaps (also moves to zero)
        else:
            design_vectors = design_vectors.copy()
            for i_dv in range(design_vectors.shape[1]):
                des_var = design_vectors[:, i_dv].copy()
                unique_values = np.sort(np.unique(des_var))
                for i_unique, value in enumerate(unique_values):
                    design_vectors[des_var == value, i_dv] = i_unique

        # Remove design variables with not enough options
        no_opts_mask = np.max(design_vectors, axis=0) == 0
        design_vectors = design_vectors[:, ~no_opts_mask]

        return design_vectors

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        """
        Encode a matrix of size n_patterns x n_src x n_tgt as discrete design variables.
        Returns the list of design vectors for each matrix in a n_patterns x n_dv array.
        Assumes values range between 0 and the number of options per design variable.
        """
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
    """Helper function that flattens matrices in the higher dimensions: n_mat x n_src x n_tgt --> n_mat x n_src*n_tgt"""
    return matrix.reshape(matrix.shape[0], np.prod(matrix.shape[1:]))
