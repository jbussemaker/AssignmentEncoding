import numpy as np
from typing import *
from assign_enc.lazy_encoding import *
from assign_enc.lazy.encodings.on_demand_base import *

__all__ = ['LazyAmountFirstEncoder', 'FlatLazyAmountEncoder', 'TotalLazyAmountEncoder', 'SourceLazyAmountEncoder',
           'SourceTargetLazyAmountEncoder', 'FlatLazyConnectionEncoder']


class LazyAmountEncoder:
    """Base class for encoding a set of src and tgt connection amounts"""

    def _get_n_src_tgt_array(self, n_src_n_tgt: List[Tuple[Tuple[int, ...], Tuple[int, ...]]]) -> np.ndarray:
        return np.array([n_src_conn+n_tgt_conn for n_src_conn, n_tgt_conn in n_src_n_tgt], dtype=int)

    def encode(self, n_src, n_tgt, n_src_n_tgt: List[Tuple[Tuple[int, ...], Tuple[int, ...]]]) -> List[DiscreteDV]:
        raise NotImplementedError

    def decode(self, vector: DesignVector) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        raise NotImplementedError


class LazyConnectionEncoder:
    """Base class for encoding connection selection"""

    def encode(self, n_src_n_tgt: List[Tuple[Tuple[int, ...], Tuple[int, ...]]], encoder: OnDemandLazyEncoder) \
            -> List[DiscreteDV]:
        raise NotImplementedError

    def decode(self, n_src_conn, n_tgt_conn, vector: DesignVector, encoder: OnDemandLazyEncoder,
               src_exists: np.ndarray, tgt_exists: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError


class LazyAmountFirstEncoder(OnDemandLazyEncoder):
    """Base class for an encoder that first defines one or more design variables for selecting the amount of
    connections, and then selects the connection pattern. Generates matrices only when needed (on demand)."""

    def __init__(self, imputer: LazyImputer, amount_encoder: LazyAmountEncoder, conn_encoder: LazyConnectionEncoder):
        super().__init__(imputer)
        self.amount_encoder = amount_encoder
        self.conn_encoder = conn_encoder
        self._n_dv_amount = None

    def _encode(self) -> List[DiscreteDV]:
        n_src_n_tgt = list(self.iter_n_src_n_tgt())
        dv_amount = self.amount_encoder.encode(self.n_src, self.tgt, n_src_n_tgt)
        self._n_dv_amount = len(dv_amount)

        dv_conn = self.conn_encoder.encode(n_src_n_tgt, self)

        return dv_amount+dv_conn

    def _decode(self, vector: DesignVector, src_exists: np.ndarray, tgt_exists: np.ndarray) -> Optional[np.ndarray]:
        amount_vector = vector[:self._n_dv_amount]
        n_src_tgt_conn = self.amount_encoder.decode(amount_vector)
        if n_src_tgt_conn is None:
            return
        n_src_conn, n_tgt_conn = n_src_tgt_conn

        conn_vector = vector[self._n_dv_amount:]
        return self.conn_encoder.decode(n_src_conn, n_tgt_conn, conn_vector, self, src_exists, tgt_exists)

    @staticmethod
    def group_by_values(values: np.ndarray) -> np.ndarray:
        """
        Get design vectors that uniquely map to different value combinations. Example:
        [[1 2],      [[0 0],
         [1 3],  -->  [0 1],
         [2 2],       [1 0],
         [3 2],       [2 0],
        """
        group_indices = np.empty(values.shape, dtype=int)
        row_mask_list = [np.ones((values.shape[0],), dtype=bool)]

        # Loop over columns
        for i_col in range(values.shape[1]):

            # Loop over current sub-divisions
            next_row_masks = []
            for row_mask in row_mask_list:
                # Loop over unique values in sub-divisions
                unique_values = np.sort(np.unique(values[row_mask, i_col]))
                for value_idx, value in enumerate(unique_values):
                    # Assign indices for each unique value
                    next_row_mask = row_mask & (values[:, i_col] == value)
                    next_row_masks.append(next_row_mask)
                    group_indices[next_row_mask, i_col] = value_idx

            row_mask_list = next_row_masks

        # Remove columns where there are no alternatives
        has_alternatives = np.any(group_indices > 0, axis=0)
        group_indices = group_indices[:, has_alternatives]

        return group_indices


class FlatLazyAmountEncoder(LazyAmountEncoder):

    def __init__(self):
        self._n_src_n_tgt = None

    def encode(self, n_src, n_tgt, n_src_n_tgt: List[Tuple[Tuple[int, ...], Tuple[int, ...]]]) -> List[DiscreteDV]:
        self._n_src_n_tgt = n_src_n_tgt
        return [DiscreteDV(n_opts=len(n_src_n_tgt))]

    def decode(self, vector: DesignVector) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        return self._n_src_n_tgt[vector[0]]


class GroupedLazyAmountEncoder(LazyAmountEncoder):

    def __init__(self):
        self._dv_val_map = None

    def encode(self, n_src, n_tgt, n_src_n_tgt: List[Tuple[Tuple[int, ...], Tuple[int, ...]]]) -> List[DiscreteDV]:
        n_src_tgt_arr = self._get_n_src_tgt_array(n_src_n_tgt)
        dv_group_values = self._get_dv_group_values(n_src, n_tgt, n_src_tgt_arr)
        dv_values = LazyAmountFirstEncoder.group_by_values(dv_group_values)
        self._dv_val_map = {tuple(dv_val): n_src_n_tgt[i] for i, dv_val in enumerate(dv_values)}
        return [DiscreteDV(n_opts=n) for n in np.max(dv_values, axis=0)+1]

    def decode(self, vector: DesignVector) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        try:
            return self._dv_val_map[tuple(vector)]
        except KeyError:
            pass

    def _get_dv_group_values(self, n_src, n_tgt, n_src_tgt_arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TotalLazyAmountEncoder(GroupedLazyAmountEncoder):

    def _get_dv_group_values(self, n_src, n_tgt, n_src_tgt_arr: np.ndarray) -> np.ndarray:
        return np.column_stack([
            np.sum(n_src_tgt_arr[:, :n_src], axis=1),
            np.arange(n_src_tgt_arr.shape[0]),
        ])


class SourceLazyAmountEncoder(GroupedLazyAmountEncoder):

    def _get_dv_group_values(self, n_src, n_tgt, n_src_tgt_arr: np.ndarray) -> np.ndarray:
        return np.column_stack([
            n_src_tgt_arr[:, :n_src],
            np.arange(n_src_tgt_arr.shape[0]),
        ])


class SourceTargetLazyAmountEncoder(GroupedLazyAmountEncoder):

    def _get_dv_group_values(self, n_src, n_tgt, n_src_tgt_arr: np.ndarray) -> np.ndarray:
        return n_src_tgt_arr


class FlatLazyConnectionEncoder(LazyConnectionEncoder):

    def encode(self, n_src_n_tgt: List[Tuple[Tuple[int, ...], Tuple[int, ...]]], encoder: OnDemandLazyEncoder) \
            -> List[DiscreteDV]:

        i_tot_max = np.argmax(np.sum(np.array([n_src_conn for n_src_conn, _ in n_src_n_tgt]), axis=1))

        n_src_conn, n_tgt_conn = n_src_n_tgt[i_tot_max]
        n_max = encoder.count_matrices(n_src_conn, n_tgt_conn)
        return [DiscreteDV(n_opts=n_max)] if n_max > 1 else []

    def decode(self, n_src_conn, n_tgt_conn, vector: DesignVector, encoder: OnDemandLazyEncoder,
               src_exists: np.ndarray, tgt_exists: np.ndarray) -> Optional[np.ndarray]:

        matrices = encoder.get_matrices(n_src_conn, n_tgt_conn, src_exists, tgt_exists)
        if len(vector) == 0:
            return matrices[0, :, :]

        i_conn = vector[0]
        if i_conn < matrices.shape[0]:
            return matrices[i_conn, :, :]
