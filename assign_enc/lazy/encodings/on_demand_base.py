import numpy as np
from typing import *
from assign_enc.lazy_encoding import *

__all__ = ['OnDemandLazyEncoder']


class OnDemandLazyEncoder(LazyEncoder):
    """Base class for a lazy encoder that generates matrices for a specific n_src, n_tgt combination only when needed."""

    def __init__(self, imputer: LazyImputer):
        super().__init__(imputer)
        self._matrix_cache = {}

    def _encode_prepare(self):
        self._matrix_cache = {}

    def iter_n_src_n_tgt(self, existence: NodeExistence = None):
        """Iterate over combinations of source and target connections amounts: n_src_conn, n_tgt_conn, existence
        (tuples of length n_src and n_tgt respectively)."""
        yield from self._matrix_gen.iter_n_sources_targets(existence=existence)

    def count_matrices(self, n_src_conn, n_tgt_conn) -> int:
        """Count the number of connection matrices for a given src and tgt connection amount.
        Note this is actually not much faster than actually creating the matrices, so it should be used carefully"""
        return self.get_matrices(n_src_conn, n_tgt_conn).shape[0]

    def get_matrices(self, n_src_conn, n_tgt_conn) -> np.ndarray:
        """Get matrices belonging to a specific combination of src and tgt connection amount"""

        cache_key = (tuple(n_src_conn), tuple(n_tgt_conn))
        if cache_key in self._matrix_cache:
            return self._matrix_cache[cache_key]

        self._matrix_cache[cache_key] = matrices = self._matrix_gen.get_matrices_by_n_conn(n_src_conn, n_tgt_conn)
        return matrices

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
