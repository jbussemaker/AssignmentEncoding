import copy
import itertools
import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *

__all__ = ['LazyImputer', 'LazyEncoder', 'DesignVector', 'DiscreteDV', 'NodeExistence']


class LazyImputer:
    """Base class for imputing design vectors to select existing matrices, works with lazy encoding."""

    def __init__(self):
        self._matrix_gen: Optional[AggregateAssignmentMatrixGenerator] = None
        self._existence_des_vars: Optional[Dict[NodeExistence, List[DiscreteDV]]] = None
        self._des_vars: Optional[List[DiscreteDV]] = None
        self._decode_func = None
        self._impute_cache = {}

    def initialize(self, matrix_gen: AggregateAssignmentMatrixGenerator,
                   existence_des_vars: Dict[NodeExistence, List[DiscreteDV]], design_vars: List[DiscreteDV],
                   decode_func):
        self._matrix_gen = matrix_gen
        self._existence_des_vars = existence_des_vars
        self._des_vars = design_vars
        self._decode_func = decode_func
        self._impute_cache = {}

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[np.ndarray]:
        return self._decode_func(vector, existence)

    def impute(self, vector: DesignVector, matrix: Optional[np.ndarray], existence: NodeExistence,
               tried_vectors: set = None) -> Tuple[DesignVector, np.ndarray]:
        """Returns the imputed design vector and associated connection matrix"""

        # Check in cache
        cache_key = hash((tuple(vector), hash(existence)))
        if cache_key in self._impute_cache:
            return self._impute_cache[cache_key]

        def _validate(mat):
            if mat is None:
                return False
            return self._matrix_gen.validate_matrix(mat, existence=existence)

        # If none of the nodes exist, return empty matrix
        n_src, n_tgt = len(self._matrix_gen.src), len(self._matrix_gen.tgt)
        if existence.none_exists(n_src, n_tgt):
            imputed_vector = [0]*len(vector)
            imputed_matrix = np.zeros((n_src, n_tgt), dtype=int)

        else:
            # Try to modify vector such that it adheres to the src and tgt exist masks
            if matrix is not None:
                matrix = matrix.copy()
                matrix[~existence.src_exists_mask(n_src), :] = 0
                matrix[:, ~existence.tgt_exists_mask(n_tgt)] = 0
            if matrix is not None and _validate(matrix):
                imputed_vector, imputed_matrix = vector, matrix

            else:  # Custom imputation
                if tried_vectors is None:
                    tried_vectors = set()
                tried_vectors.add(tuple(vector))

                vector = np.array(vector)
                imputed_vector, imputed_matrix = self._impute(vector, matrix, existence, _validate, tried_vectors)

        # Update cache
        self._impute_cache[cache_key] = (imputed_vector, imputed_matrix)
        return imputed_vector, imputed_matrix

    def _impute(self, vector: DesignVector, matrix: Optional[np.ndarray], existence: NodeExistence,
                validate: Callable[[np.ndarray], bool], tried_vectors: Set[Tuple[int, ...]]) \
            -> Tuple[DesignVector, np.ndarray]:
        """Returns the imputed design vector and associated connection matrix"""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class LazyEncoder(Encoder):
    """Encoder that skips the matrix-generation step (so it might be better suited for large numbers of connections) by
    relying on two-way design variable encoders."""

    _n_mc_imputation_ratio = 10000

    def __init__(self, imputer: LazyImputer):
        self._matrix_gen: Optional[AggregateAssignmentMatrixGenerator] = None
        self._existence_design_vars: Optional[Dict[NodeExistence, List[DiscreteDV]]] = None
        self._design_vars: Optional[List[DiscreteDV]] = None
        self._imputer = imputer

    def set_nodes(self, src: List[Node], tgt: List[Node], excluded: List[Tuple[Node, Node]] = None,
                  existence_patterns: NodeExistencePatterns = None):
        self._matrix_gen = gen = AggregateAssignmentMatrixGenerator(
            src, tgt, excluded=excluded, existence_patterns=existence_patterns)

        self._encode_prepare()
        self._existence_design_vars = existence_dvs = \
            {existence: self._encode(existence) for existence in gen.iter_existence()}
        self._design_vars = dvs = self._merge_design_vars(list(existence_dvs.values()))
        for i, dv in enumerate(dvs):
            if dv.n_opts < 2:
                raise RuntimeError(f'All design variables must have at least 2 options: {i} has {dv.n_opts} opts')

        self._imputer.initialize(self._matrix_gen, self._existence_design_vars, self._design_vars, self._decode)

    def set_imputer(self, imputer: LazyImputer):
        self._imputer = imputer
        if self._matrix_gen is not None:
            self._imputer.initialize(self._matrix_gen, self._existence_design_vars, self._design_vars, self._decode)

    def get_for_imputer(self, imputer: LazyImputer):
        encoder: LazyEncoder = copy.deepcopy(self)
        encoder.set_imputer(imputer)
        return encoder

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
    def existence_patterns(self):
        return self._matrix_gen.existence_patterns

    @property
    def matrix_gen(self) -> AggregateAssignmentMatrixGenerator:
        return self._matrix_gen

    @property
    def design_vars(self) -> List[DiscreteDV]:
        return self._design_vars

    def get_imputation_ratio(self, use_real_matrix=True) -> float:
        if use_real_matrix:
            n_matrix = self._matrix_gen.count_all_matrices(max_by_existence=False)
            return self.get_n_design_points()/n_matrix

        n_sample = self._n_mc_imputation_ratio
        n_valid = 0

        if n_sample > self.get_n_design_points():
            # Exhaustive sampling
            n_sample = 0
            for dv in itertools.product(*[list(range(dv.n_opts)) for dv in self.design_vars]):
                n_sample += 1
                if self.is_valid_vector(list(dv)):
                    n_valid += 1

        else:  # Monte Carlo sampling
            for _ in range(n_sample):
                if self.is_valid_vector(self.get_random_design_vector()):
                    n_valid += 1

        return n_sample/n_valid

    def _correct_vector_size(self, vector: DesignVector) -> Tuple[DesignVector, int, int]:
        n_dv = len(self.design_vars)
        vector, n_extra = EagerEncoder.correct_vector_size(n_dv, vector)
        return vector, n_dv, n_extra

    def get_matrix(self, vector: DesignVector, existence: NodeExistence = None) -> Tuple[DesignVector, np.ndarray]:
        """Select a connection matrix (n_src x n_tgt) and impute the design vector if needed."""

        vector, n_dv, n_extra = self._correct_vector_size(vector)
        extra_vector = [0]*n_extra
        vector, _ = EagerEncoder.correct_vector_bounds(vector, self.design_vars)

        # Decode matrix
        matrix, existence = self._decode_vector(vector, existence)

        # Validate matrix
        if matrix is not None and self._matrix_gen.validate_matrix(matrix, existence=existence):
            return list(vector)+extra_vector, matrix

        vector, matrix = self._imputer.impute(vector, matrix, existence)
        return list(vector)+extra_vector, matrix

    def is_valid_vector(self, vector: DesignVector, existence: NodeExistence = None):

        original_vector = vector
        vector, n_dv, n_extra = self._correct_vector_size(vector)
        vector, is_corrected = EagerEncoder.correct_vector_bounds(vector, self.design_vars)
        if is_corrected:
            return False

        matrix, existence = self._decode_vector(vector, existence)
        if matrix is None:
            return False

        if not self._matrix_gen.validate_matrix(matrix, existence=existence):
            return False

        if n_extra > 0 and sum(original_vector[n_dv:]) != 0:
            return False
        return True

    def _decode_vector(self, vector: DesignVector, existence: NodeExistence = None):
        if existence is None:
            existence = NodeExistence()
        matrix = self._decode(vector, existence=existence)
        return matrix, existence

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

    @staticmethod
    def _filter_dvs(dvs: List[DiscreteDV]) -> Tuple[List[DiscreteDV], np.ndarray, int]:
        """Return design variables that have at least 2 options"""
        i_valid_dv = []
        valid_dv = []
        for i, dv in enumerate(dvs):
            if dv.n_opts >= 2:
                i_valid_dv.append(i)
                valid_dv.append(dv)

        i_valid_dv = np.array(i_valid_dv, dtype=int)
        return valid_dv, i_valid_dv, len(dvs)

    @staticmethod
    def _unfilter_dvs(vector: DesignVector, i_valid_dv: np.ndarray, n: int) -> Optional[DesignVector]:
        all_vector = np.zeros((n,), dtype=int)
        all_vector[i_valid_dv] = vector[:n]

        # Ensure that all further design variables are zero to avoid multiple design vectors mapping to the same design
        if n < len(vector) and sum(vector[n:]) != 0:
            return

        return all_vector

    @staticmethod
    def _merge_design_vars(design_vars_list: List[List[DiscreteDV]]) -> List[DiscreteDV]:
        return EagerEncoder.merge_design_vars(design_vars_list)

    def _encode_prepare(self):
        pass

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        """Encode the assignment problem (given by src and tgt nodes) directly to design variables"""
        raise NotImplementedError

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[np.ndarray]:
        """Return the connection matrix as would be encoded by the given design vector"""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
