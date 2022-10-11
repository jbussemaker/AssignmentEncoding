import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *

__all__ = ['LazyImputer', 'LazyEncoder', 'DesignVector', 'DiscreteDV']


class LazyImputer:
    """Base class for imputing design vectors to select existing matrices, works with lazy encoding."""

    def __init__(self):
        self._matrix_gen: Optional[AggregateAssignmentMatrixGenerator] = None
        self._des_vars: Optional[List[DiscreteDV]] = None
        self._decode_func = None
        self._impute_cache = {}

    def initialize(self, matrix_gen: AggregateAssignmentMatrixGenerator, design_vars: List[DiscreteDV], decode_func):
        self._matrix_gen = matrix_gen
        self._des_vars = design_vars
        self._decode_func = decode_func
        self._impute_cache = {}

    def _decode(self, vector: DesignVector, src_exists: np.ndarray, tgt_exists: np.ndarray) -> np.ndarray:
        return self._decode_func(vector, src_exists, tgt_exists)

    def impute(self, vector: DesignVector, matrix: np.ndarray, src_exists: np.ndarray, tgt_exists: np.ndarray,
               tried_vectors: set = None) -> Tuple[DesignVector, np.ndarray]:
        """Returns the imputed design vector and associated connection matrix"""

        # Check in cache
        cache_key = hash((tuple(vector), tuple(src_exists), tuple(tgt_exists)))
        if cache_key in self._impute_cache:
            return self._impute_cache[cache_key]

        def _validate(mat):
            return self._matrix_gen.validate_matrix(mat, src_exists=src_exists, tgt_exists=tgt_exists)

        # If none of the nodes exist, return empty matrix
        if np.all(~src_exists) and np.all(~tgt_exists):
            imputed_vector = [0]*len(vector)
            imputed_matrix = matrix*0

        else:
            # Try to modify vector such that it adheres to the src and tgt exist masks
            matrix = matrix.copy()
            matrix[~src_exists, :] = 0
            matrix[:, ~tgt_exists] = 0
            if _validate(matrix):
                imputed_vector, imputed_matrix = vector, matrix

            else:  # Custom imputation
                if tried_vectors is None:
                    tried_vectors = set()
                tried_vectors.add(tuple(vector))

                vector = np.array(vector)
                imputed_vector, imputed_matrix = \
                    self._impute(vector, matrix, src_exists, tgt_exists, _validate, tried_vectors)

        # Update cache
        self._impute_cache[cache_key] = (imputed_vector, imputed_matrix)
        return imputed_vector, imputed_matrix

    def _impute(self, vector: DesignVector, matrix: np.ndarray, src_exists: np.ndarray, tgt_exists: np.ndarray,
                validate: Callable[[np.ndarray], bool], tried_vectors: Set[Tuple[int, ...]]) \
            -> Tuple[DesignVector, np.ndarray]:
        """Returns the imputed design vector and associated connection matrix"""
        raise NotImplementedError


class LazyEncoder:
    """Encoder that skips the matrix-generation step (so it might be better suited for large numbers of connections) by
    relying on two-way design variable encoders."""

    def __init__(self, imputer: LazyImputer):
        self._matrix_gen: Optional[AggregateAssignmentMatrixGenerator] = None
        self._design_vars: Optional[List[DiscreteDV]] = None
        self._imputer = imputer

    def set_nodes(self, src: List[Node], tgt: List[Node], excluded: List[Tuple[Node, Node]] = None):
        self._matrix_gen = AggregateAssignmentMatrixGenerator(src, tgt, excluded=excluded)
        self._design_vars = self._encode()
        self._imputer.initialize(self._matrix_gen, self._design_vars, self._decode)

    @property
    def src(self):
        return self._matrix_gen.src

    @property
    def tgt(self):
        return self._matrix_gen.tgt

    @property
    def n_src(self):
        return len(self.src)

    @property
    def n_tgt(self):
        return len(self.tgt)

    @property
    def ex(self):
        return self._matrix_gen.ex

    @property
    def design_vars(self) -> List[DiscreteDV]:
        return self._design_vars

    def get_random_design_vector(self) -> DesignVector:
        return [dv.get_random() for dv in self.design_vars]

    def get_matrix(self, vector: DesignVector, src_exists: List[bool] = None, tgt_exists: List[bool] = None) \
            -> Tuple[DesignVector, np.ndarray]:
        """Select a connection matrix (n_src x n_tgt) and impute the design vector if needed."""

        # Decode matrix
        matrix, src_exists, tgt_exists = self._decode_vector(vector, src_exists, tgt_exists)

        # Validate matrix
        if self._matrix_gen.validate_matrix(matrix, src_exists=src_exists, tgt_exists=tgt_exists):
            return vector, matrix

        return self._imputer.impute(vector, matrix, src_exists, tgt_exists)

    def is_valid_vector(self, vector: DesignVector, src_exists: List[bool] = None, tgt_exists: List[bool] = None):
        matrix, src_exists, tgt_exists = self._decode_vector(vector, src_exists, tgt_exists)
        return self._matrix_gen.validate_matrix(matrix, src_exists=src_exists, tgt_exists=tgt_exists)

    def _decode_vector(self, vector: DesignVector, src_exists: List[bool] = None, tgt_exists: List[bool] = None):
        src_exists = np.array([True for _ in range(self.n_src)] if src_exists is None else src_exists)
        tgt_exists = np.array([True for _ in range(self.n_tgt)] if tgt_exists is None else tgt_exists)
        matrix = self._decode(vector, src_exists=src_exists, tgt_exists=tgt_exists)
        return matrix, src_exists, tgt_exists

    def get_conn_idx(self, matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Convert matrix to edge tuples"""
        edges = []
        for i_src in range(matrix.shape[0]):
            for j_tgt in range(matrix.shape[1]):
                for _ in range(matrix[i_src, j_tgt]):
                    edges.append((i_src, j_tgt))
        return edges

    def get_conns(self, matrix: np.ndarray) -> List[Tuple[Node, Node]]:
        """Convert matrix to edge tuples"""
        edges = []
        for i_src in range(matrix.shape[0]):
            src = self.src[i_src]
            for j_tgt in range(matrix.shape[1]):
                tgt = self.tgt[j_tgt]
                for _ in range(matrix[i_src, j_tgt]):
                    edges.append((src, tgt))
        return edges

    def _encode(self) -> List[DiscreteDV]:
        """Encode the assignment problem (given by src and tgt nodes) directly to design variables"""
        raise NotImplementedError

    def _decode(self, vector: DesignVector, src_exists: np.ndarray, tgt_exists: np.ndarray) -> np.ndarray:
        """Return the connection matrix as would be encoded by the given design vector"""
        raise NotImplementedError
