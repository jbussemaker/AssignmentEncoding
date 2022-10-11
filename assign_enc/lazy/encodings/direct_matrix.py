import numpy as np
from typing import *
from assign_enc.lazy_encoding import *

__all__ = ['LazyDirectMatrixEncoder']


class LazyDirectMatrixEncoder(LazyEncoder):
    """
    Defines one design variable for each matrix element.
    Uses the following logic to determine the amount of maximum connections per source-target pair:
    - If the connection is excluded, max connections is 0
    - If for either the source or target repeat_allowed is False, max connections is 1
    - Otherwise, the max connections is min(source_max, target_max, overall_max)

    The minimum number of connections is always assumed to be 0.
    If the nr of max connections is 0, the design variable is skipped.
    """

    def __init__(self, imputer: LazyImputer):
        self._dv_idx_map = []
        super().__init__(imputer)

    def _encode(self) -> List[DiscreteDV]:
        matrix_gen = self._matrix_gen
        overall_max = matrix_gen.max_conn
        blocked_mask = matrix_gen.conn_blocked_mask
        no_repeat_mask = matrix_gen.no_repeat_mask

        dvs = []
        self._dv_idx_map = dv_idx_map = []
        src, tgt = self.src, self.tgt
        for i in range(self.n_src):
            src_max = np.inf if src[i].max_inf else src[i].conns[-1]
            for j in range(self.n_tgt):
                # Check if connection is blocked
                if blocked_mask[i, j]:
                    continue

                # Get maximum number of connections
                tgt_max = np.inf if tgt[j].max_inf else tgt[j].conns[-1]
                max_conn = min(src_max, tgt_max, overall_max)

                # Constrain to 1 if repetition is not allowed
                if max_conn > 1 and no_repeat_mask[i, j]:
                    max_conn = 1

                # Define design variable
                if max_conn >= 1:
                    dvs.append(DiscreteDV(n_opts=max_conn+1))
                    dv_idx_map.append((i, j))

        return dvs

    def _decode(self, vector: DesignVector, src_exists: np.ndarray, tgt_exists: np.ndarray) -> np.ndarray:
        matrix = np.zeros((self.n_src, self.n_tgt), dtype=int)
        for i_dv, (i, j) in enumerate(self._dv_idx_map):
            matrix[i, j] = vector[i_dv]

        return matrix
