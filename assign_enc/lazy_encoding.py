import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *

__all__ = ['LazyImputer', 'LazyEncoder']


class LazyImputer:
    """Base class for imputing design vectors to select existing matrices, works with lazy encoding."""


class LazyEncoder:
    """Encoder that skips the matrix-generation step (so it might be better suited for large numbers of connections) by
    relying on two-way design variable encoders."""

    def __init__(self):
        self.src = None
        self.tgt = None
        self._ex = None

    def set_nodes(self, src: List[Node], tgt: List[Node], excluded: List[Tuple[Node, Node]] = None):
        self.src = src
        self.tgt = tgt
        self.ex = excluded
        self._encode()

    @property
    def ex(self) -> Optional[Set[Tuple[int, int]]]:
        return self._ex

    @ex.setter
    def ex(self, excluded: Optional[List[Tuple[int, int]]]):
        if excluded is None:
            self._ex = None
        else:
            src_idx = {src: i for i, src in enumerate(self.src)}
            tgt_idx = {tgt: i for i, tgt in enumerate(self.tgt)}
            self._ex = set([(src_idx[ex[0]], tgt_idx[ex[1]]) for ex in excluded])
            if len(self._ex) == 0:
                self._ex = None

    @property
    def design_vars(self) -> List[DiscreteDV]:
        raise NotImplementedError

    def get_random_design_vector(self) -> DesignVector:
        return [dv.get_random() for dv in self.design_vars]

    def get_matrix(self, vector: DesignVector, src_exists: List[bool] = None, tgt_exists: List[bool] = None) \
            -> Tuple[DesignVector, np.ndarray]:
        """Select a connection matrix (n_src x n_tgt) and impute the design vector if needed."""
        raise NotImplementedError

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

    def _encode(self):
        raise NotImplementedError
