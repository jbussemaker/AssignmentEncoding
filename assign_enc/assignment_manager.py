import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_enc.lazy_encoding import *

__all__ = ['AssignmentManagerBase', 'AssignmentManager', 'LazyAssignmentManager']


class AssignmentManagerBase:
    """Base class for managing the encoding of assignment problems."""

    def get_random_design_vector(self) -> DesignVector:
        return [dv.get_random() for dv in self.design_vars]

    @property
    def design_vars(self):
        raise NotImplementedError

    def get_imputation_ratio(self) -> float:
        """Ratio of the total design space size to the actual amount of possible connections"""
        raise NotImplementedError

    def correct_vector(self, vector: DesignVector, existence: NodeExistence = None) -> DesignVector:
        """Correct the design vector so that it matches an existing connection pattern"""
        raise NotImplementedError

    def get_matrix(self, vector: DesignVector, existence: NodeExistence = None) -> Tuple[DesignVector, np.ndarray]:
        """Get connection matrix from a given design vector"""
        raise NotImplementedError

    def get_conn_idx(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, List[Tuple[int, int]]]:
        """Get node connections for a given design vector"""
        raise NotImplementedError

    def get_conns(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, List[Tuple[Node, Node]]]:
        """Get node connections for a given design vector"""
        raise NotImplementedError


class AssignmentManager(AssignmentManagerBase):
    """
    Uses the matrix generator and encoder to implement the conversion between design vector and connection pattern.
    """

    def __init__(self, src: List[Node], tgt: List[Node], encoder: EagerEncoder,
                 excluded: List[Tuple[Node, Node]] = None, existence_patterns: NodeExistencePatterns = None,
                 cache=True):
        self._matrix_gen = gen = AggregateAssignmentMatrixGenerator(
            src, tgt, excluded=excluded, existence_patterns=existence_patterns)
        self._encoder = encoder

        encoder.matrix = gen.get_agg_matrix(cache=cache)

    @property
    def matrix_gen(self) -> AggregateAssignmentMatrixGenerator:
        return self._matrix_gen

    @property
    def encoder(self) -> EagerEncoder:
        return self._encoder

    @property
    def matrix(self) -> Dict[NodeExistence, np.ndarray]:
        return self._encoder.matrix

    @property
    def design_vars(self):
        return self._encoder.design_vars

    def get_imputation_ratio(self) -> float:
        return self._encoder.get_imputation_ratio()

    def correct_vector(self, vector: DesignVector, existence: NodeExistence = None) -> DesignVector:
        """Correct the design vector so that it matches an existing connection pattern"""
        imputed_vector, _ = self._encoder.get_matrix(vector, existence=existence)
        return imputed_vector

    def get_matrix(self, vector: DesignVector, existence: NodeExistence = None) -> Tuple[DesignVector, np.ndarray]:
        """Get connection matrix from a given design vector"""
        return self._encoder.get_matrix(vector, existence=existence)

    def get_conn_idx(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, List[Tuple[int, int]]]:
        """Get node connections for a given design vector"""

        # Get matrix and impute design vector
        imputed_vector, matrix = self.get_matrix(vector, existence=existence)

        # Get connections from matrix
        edges_idx = self._matrix_gen.get_conn_idx(matrix)

        return imputed_vector, edges_idx

    def get_conns(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, List[Tuple[Node, Node]]]:
        """Get node connections for a given design vector"""

        # Get matrix and impute design vector
        imputed_vector, matrix = self.get_matrix(vector, existence=existence)

        # Get connections from matrix
        edges = self._matrix_gen.get_conns(matrix)

        return imputed_vector, edges


class LazyAssignmentManager(AssignmentManagerBase):

    def __init__(self, src: List[Node], tgt: List[Node], encoder: LazyEncoder,
                 excluded: List[Tuple[Node, Node]] = None, existence_patterns: NodeExistencePatterns = None):
        self._encoder = encoder
        encoder.set_nodes(src, tgt, excluded=excluded, existence_patterns=existence_patterns)

    @property
    def design_vars(self):
        return self._encoder.design_vars

    def get_imputation_ratio(self) -> float:
        return self._encoder.get_imputation_ratio()

    def correct_vector(self, vector: DesignVector, existence: NodeExistence = None) -> DesignVector:
        """Correct the design vector so that it matches an existing connection pattern"""
        imputed_vector, _ = self._encoder.get_matrix(vector, existence=existence)
        return imputed_vector

    def get_matrix(self, vector: DesignVector, existence: NodeExistence = None) -> Tuple[DesignVector, np.ndarray]:
        """Get connection matrix from a given design vector"""
        return self._encoder.get_matrix(vector, existence=existence)

    def get_conn_idx(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, List[Tuple[int, int]]]:
        """Get node connections for a given design vector"""
        imputed_vector, matrix = self._encoder.get_matrix(vector, existence=existence)

        # Get connections from matrix
        edges_idx = self._encoder.get_conn_idx(matrix)
        return imputed_vector, edges_idx

    def get_conns(self, vector: DesignVector, existence: NodeExistence = None) \
            -> Tuple[DesignVector, List[Tuple[Node, Node]]]:
        """Get node connections for a given design vector"""
        imputed_vector, matrix = self._encoder.get_matrix(vector, existence=existence)

        # Get connections from matrix
        edges = self._encoder.get_conns(matrix)
        return imputed_vector, edges
