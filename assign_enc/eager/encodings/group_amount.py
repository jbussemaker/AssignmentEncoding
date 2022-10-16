import itertools
import numpy as np
from assign_enc.encoding import *
from assign_enc.eager.encodings.grouped_base import *

__all__ = ['AmountGrouper', 'LocationGrouper', 'AmountFirstGroupedEncoder', 'TotalAmountGrouper', 'SourceAmountGrouper',
           'SourceAmountFlattenedGrouper', 'TargetAmountGrouper', 'TargetAmountFlattenedGrouper',
           'OneVarLocationGrouper', 'FlatIndexLocationGrouper', 'RelFlatIndexLocationGrouper',
           'CoordIndexLocationGrouper', 'RelCoordIndexLocationGrouper']


class AmountGrouper:
    """Base class for grouping by total connection amount."""

    def get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LocationGrouper:
    """Base class for grouping by connection location."""

    def get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class AmountFirstGroupedEncoder(GroupedEncoder):
    """Grouped encoder that first groups by the total amount of connections to make, and then by the locations of these
    connections."""

    def __init__(self, imputer, amount_grouper: AmountGrouper, loc_grouper: LocationGrouper, **kwargs):
        self.amount_grouper = amount_grouper
        self.loc_grouper = loc_grouper
        super().__init__(imputer, **kwargs)

    def _get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        return np.column_stack([
            self.amount_grouper.get_grouping_values(matrix),
            self.loc_grouper.get_grouping_values(matrix),
        ])


class TotalAmountGrouper(AmountGrouper):
    """Group by total amount of connections"""

    def get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        return np.sum(flatten_matrix(matrix), axis=1)


class SourceAmountGrouper(AmountGrouper):
    """Group by number of connections from source nodes"""

    def get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        return np.sum(matrix, axis=2)


class SourceAmountFlattenedGrouper(AmountGrouper):
    """Group by number of connections from source nodes; summarized in one design variable"""

    def get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        n_src = np.sum(matrix, axis=2)
        _, unique_indices = np.unique(n_src, return_inverse=True, axis=0)
        return np.array([unique_indices]).T


class TargetAmountGrouper(AmountGrouper):
    """Group by number of connections to target nodes"""

    def get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        return np.sum(matrix, axis=1)


class TargetAmountFlattenedGrouper(AmountGrouper):
    """Group by number of connections to target nodes; summarized in one design variable"""

    def get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        n_src = np.sum(matrix, axis=1)
        _, unique_indices = np.unique(n_src, return_inverse=True, axis=0)
        return np.array([unique_indices]).T


class OneVarLocationGrouper(LocationGrouper):
    """One design variable for each remaining matrix after selecting the amounts"""

    def get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        return np.arange(matrix.shape[0])


class FlatIndexLocationGrouper(LocationGrouper):
    """Group by connection location indices"""

    def get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        matrix_flat = flatten_matrix(matrix)
        n_conn_max = np.max(np.sum(matrix_flat, axis=1))

        n_mat = matrix.shape[0]
        dv_loc_idx = np.zeros((n_mat, n_conn_max), dtype=int)
        for i_mat in range(n_mat):
            loc_idx = self._get_loc_indices(matrix_flat[i_mat, :])
            dv_loc_idx[i_mat, :len(loc_idx)] = loc_idx
        return dv_loc_idx

    @classmethod
    def _get_loc_indices(cls, conn_arr: np.ndarray) -> np.ndarray:
        conn_arr = conn_arr.copy()
        loc_idx = []
        idx = 0
        while idx < len(conn_arr):
            if conn_arr[idx] > 0:
                loc_idx.append(idx)
                conn_arr[idx] -= 1
            else:
                idx += 1
        return np.array(loc_idx)


class RelFlatIndexLocationGrouper(FlatIndexLocationGrouper):
    """Group by relative connection location indices"""

    @classmethod
    def _get_loc_indices(cls, conn_arr: np.ndarray) -> np.ndarray:
        abs_loc_idx = super()._get_loc_indices(conn_arr)
        return np.diff(np.concatenate([[0], abs_loc_idx]))


class CoordIndexLocationGrouper(LocationGrouper):
    """Group by connection location indices (encoded as i_src, i_tgt)"""

    def get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        matrix_flat = flatten_matrix(matrix)
        n_conn_max = np.max(np.sum(matrix_flat, axis=1))*2

        n_mat, n_src, n_tgt = matrix.shape[0], matrix.shape[1], matrix.shape[2]
        dv_loc_idx = np.zeros((n_mat, n_conn_max), dtype=int)
        idx_map = list(itertools.product(range(n_src), range(n_tgt)))
        for i_mat in range(n_mat):
            loc_idx = self._get_loc_indices(matrix_flat[i_mat, :], idx_map)
            dv_loc_idx[i_mat, :len(loc_idx)] = loc_idx
        return dv_loc_idx

    @classmethod
    def _get_loc_indices(cls, conn_arr: np.ndarray, idx_map) -> np.ndarray:
        conn_arr = conn_arr.copy()
        loc_idx = []
        idx = 0
        while idx < len(conn_arr):
            if conn_arr[idx] > 0:
                loc_idx.append(idx_map[idx])
                conn_arr[idx] -= 1
            else:
                idx += 1
        return np.array([idx for (i_src, i_tgt) in loc_idx for idx in [i_src, i_tgt]])


class RelCoordIndexLocationGrouper(CoordIndexLocationGrouper):
    """Group by relative connection location indices (encoded as i_src, i_tgt)"""

    @classmethod
    def _get_loc_indices(cls, conn_arr: np.ndarray, idx_map) -> np.ndarray:
        abs_loc_idx = super()._get_loc_indices(conn_arr, idx_map)
        rel_loc_idx = np.zeros(abs_loc_idx.shape)
        rel_loc_idx[0::2] = np.diff(np.concatenate([[0], abs_loc_idx[0::2]]))
        rel_loc_idx[1::2] = np.diff(np.concatenate([[0], abs_loc_idx[1::2]]))
        return rel_loc_idx
