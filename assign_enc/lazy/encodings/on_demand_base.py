import numpy as np
from typing import *
from assign_enc.lazy_encoding import *

__all__ = ['OnDemandLazyEncoder']


class OnDemandLazyEncoder(LazyEncoder):
    """Base class for a lazy encoder that generates matrices for a specific n_src, n_tgt combination only when needed."""

    def __init__(self, imputer: LazyImputer):
        super().__init__(imputer)
        self._matrix_cache = {}
        self._filtered_matrix_cache = {}

    def set_nodes(self, *args, **kwargs):
        super().set_nodes(*args, **kwargs)
        self._matrix_cache = {}
        self._filtered_matrix_cache = {}

    def iter_n_src_n_tgt(self):
        """Iterate over combinations of source and target connections amounts: n_src_conn, n_tgt_conn
        (tuples of length n_src and n_tgt respectively)."""
        yield from self._matrix_gen.iter_n_sources_targets()

    def count_matrices(self, n_src_conn, n_tgt_conn) -> int:
        """Count the number of connection matrices for a given src and tgt connection amount.
        Note this is actually not much faster than actually creating the matrices, so it should be used carefully"""
        return self.get_matrices(n_src_conn, n_tgt_conn).shape[0]

    def get_matrices(self, n_src_conn, n_tgt_conn, src_exists: np.ndarray = None, tgt_exists: np.ndarray = None) \
            -> np.ndarray:
        """Get matrices belonging to a specific combination of src and tgt connection amount"""

        # Generate matrices if not already done
        cache_key = (tuple(n_src_conn), tuple(n_tgt_conn))
        if cache_key in self._matrix_cache:
            matrices = self._matrix_cache[cache_key]
        else:
            self._matrix_cache[cache_key] = matrices = self._matrix_gen.get_matrices_by_n_conn(n_src_conn, n_tgt_conn)

        # Filter by src or tgt node existence
        cache_key_src_tgt = (cache_key, tuple(src_exists if src_exists is not None else []),
                             tuple(tgt_exists if tgt_exists is not None else []))
        if cache_key_src_tgt in self._filtered_matrix_cache:
            return self._filtered_matrix_cache[cache_key_src_tgt]

        matrix_mask = self._matrix_gen.filter_matrices(matrices, src_exists=src_exists, tgt_exists=tgt_exists)
        self._filtered_matrix_cache[cache_key_src_tgt] = filtered_matrices = matrices[matrix_mask, :, :]
        return filtered_matrices

    def _encode(self) -> List[DiscreteDV]:
        """Encode the assignment problem (given by src and tgt nodes) directly to design variables"""
        raise NotImplementedError

    def _decode(self, vector: DesignVector, src_exists: np.ndarray, tgt_exists: np.ndarray) -> Optional[np.ndarray]:
        """Return the connection matrix as would be encoded by the given design vector"""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
