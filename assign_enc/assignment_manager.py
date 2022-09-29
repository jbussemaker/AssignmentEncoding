import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *

__all__ = ['AssignmentManager']


class AssignmentManager:
    """
    Uses the matrix generator and encoder to implement the conversion between design vector and connection pattern.
    """

    def __init__(self, src: List[Node], tgt: List[Node], encoder: Encoder, excluded: List[Tuple[Node, Node]] = None):
        self._matrix_gen = gen = AggregateAssignmentMatrixGenerator(src, tgt, excluded=excluded)
        self._encoder = encoder

        encoder.matrix = gen.get_agg_matrix()

    @property
    def matrix_gen(self) -> AggregateAssignmentMatrixGenerator:
        return self._matrix_gen

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @property
    def matrix(self) -> np.ndarray:
        return self._encoder.matrix

    @property
    def design_vars(self):
        return self._encoder.design_vars

    def get_random_design_vector(self) -> DesignVector:
        return self._encoder.get_random_design_vector()

    def correct_vector(self, vector: DesignVector, src_exists: List[bool] = None, tgt_exists: List[bool] = None) \
            -> DesignVector:
        """Correct the design vector so that it matches an existing connection pattern"""

        # Pre-filter matrices if (potentially) not all nodes exist
        matrix_mask = None
        if src_exists is not None or tgt_exists is not None:
            matrix_mask = self._matrix_gen.filter_matrices(self.matrix, src_exists=src_exists, tgt_exists=tgt_exists)

        # Get matrix and impute design vector
        imputed_vector, _ = self._encoder.get_matrix(vector, matrix_mask=matrix_mask)
        return imputed_vector

    def get_conns(self, vector: DesignVector, src_exists: List[bool] = None, tgt_exists: List[bool] = None) \
            -> Tuple[DesignVector, List[Tuple[Node, Node]]]:
        """Get node connections for a given design vector"""

        # Pre-filter matrices if (potentially) not all nodes exist
        matrix_mask = None
        if src_exists is not None or tgt_exists is not None:
            matrix_mask = self._matrix_gen.filter_matrices(self.matrix, src_exists=src_exists, tgt_exists=tgt_exists)

        # Get matrix and impute design vector
        imputed_vector, matrix = self._encoder.get_matrix(vector, matrix_mask=matrix_mask)

        # Get connections from matrix
        edges = self._matrix_gen.get_conns(matrix)

        return imputed_vector, edges
