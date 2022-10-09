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

    def prepare_grouping(self, matrix: np.ndarray):
        pass

    def get_n_dvs(self, matrix: np.ndarray) -> int:
        raise NotImplementedError

    def get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LocationGrouper:
    """Base class for grouping by connection location."""

    def prepare_grouping(self, matrix: np.ndarray):
        pass

    def get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class AmountFirstGroupedEncoder(GroupedEncoder):
    """Grouped encoder that first groups by the total amount of connections to make, and then by the locations of these
    connections."""

    def __init__(self, imputer, amount_grouper: AmountGrouper, loc_grouper: LocationGrouper, **kwargs):
        self.amount_grouper = amount_grouper
        self.loc_grouper = loc_grouper
        self._n_dv_amount = None
        super().__init__(imputer, **kwargs)

    def _prepare_grouping(self, matrix: np.ndarray):
        self.amount_grouper.prepare_grouping(matrix)
        self.loc_grouper.prepare_grouping(matrix)

        self._n_dv_amount = self.amount_grouper.get_n_dvs(matrix)

    def _get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        n_dv = self._n_dv_amount
        if dv_idx >= n_dv:
            return self.loc_grouper.get_grouping_criteria(dv_idx-n_dv, sub_matrix, i_sub_matrix)
        return self.amount_grouper.get_grouping_criteria(dv_idx, sub_matrix, i_sub_matrix)


class TotalAmountGrouper(AmountGrouper):
    """Group by total amount of connections"""

    def get_n_dvs(self, matrix: np.ndarray) -> int:
        return 1

    def get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        return np.sum(flatten_matrix(sub_matrix), axis=1)


class SourceAmountGrouper(AmountGrouper):
    """Group by number of connections from source nodes"""

    def get_n_dvs(self, matrix: np.ndarray) -> int:
        return matrix.shape[1]

    def get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        return np.sum(sub_matrix[:, dv_idx, :], axis=1)


class SourceAmountFlattenedGrouper(AmountGrouper):
    """Group by number of connections from source nodes; summarized in one design variable"""

    def get_n_dvs(self, matrix: np.ndarray) -> int:
        return 1

    def get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        n_src_conn = np.sum(sub_matrix, axis=2)
        unique_n_src_conn = np.unique(n_src_conn, axis=0)

        grouping_values = np.zeros((sub_matrix.shape[0],), dtype=int)
        for i_n_src in range(unique_n_src_conn.shape[0]):
            grouping_values[np.all(n_src_conn == unique_n_src_conn[i_n_src, :], axis=1)] = i_n_src

        return grouping_values


class TargetAmountGrouper(AmountGrouper):
    """Group by number of connections to target nodes"""

    def get_n_dvs(self, matrix: np.ndarray) -> int:
        return matrix.shape[2]

    def get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        return np.sum(sub_matrix[:, :, dv_idx], axis=1)


class TargetAmountFlattenedGrouper(AmountGrouper):
    """Group by number of connections to target nodes; summarized in one design variable"""

    def get_n_dvs(self, matrix: np.ndarray) -> int:
        return 1

    def get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        n_src_conn = np.sum(sub_matrix, axis=1)
        unique_n_src_conn = np.unique(n_src_conn, axis=0)

        grouping_values = np.zeros((sub_matrix.shape[0],), dtype=int)
        for i_n_src in range(unique_n_src_conn.shape[0]):
            grouping_values[np.all(n_src_conn == unique_n_src_conn[i_n_src, :], axis=1)] = i_n_src

        return grouping_values


class OneVarLocationGrouper(LocationGrouper):
    """One design variable for each remaining matrix after selecting the amounts"""

    def get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        return np.arange(0, sub_matrix.shape[0])


class FlatIndexLocationGrouper(LocationGrouper):
    """Group by connection location indices"""

    def __init__(self):
        super().__init__()
        self._prepared_dvs = None

    def prepare_grouping(self, matrix: np.ndarray):
        matrix_flat = flatten_matrix(matrix)
        n_conn_max = np.max(np.sum(matrix_flat, axis=1))

        n_mat = matrix.shape[0]
        self._prepared_dvs = dv_loc_idx = np.zeros((n_mat, n_conn_max), dtype=int)
        for i_mat in range(n_mat):
            loc_idx = self._get_loc_indices(matrix_flat[i_mat, :])
            dv_loc_idx[i_mat, :len(loc_idx)] = loc_idx

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

    def get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        if dv_idx >= self._prepared_dvs.shape[1]:
            return np.zeros((sub_matrix.shape[0],))
        return self._prepared_dvs[i_sub_matrix, dv_idx]


class RelFlatIndexLocationGrouper(FlatIndexLocationGrouper):
    """Group by relative connection location indices"""

    @classmethod
    def _get_loc_indices(cls, conn_arr: np.ndarray) -> np.ndarray:
        abs_loc_idx = super()._get_loc_indices(conn_arr)
        return np.diff(np.concatenate([[0], abs_loc_idx]))


class CoordIndexLocationGrouper(LocationGrouper):
    """Group by connection location indices (encoded as i_src, i_tgt)"""

    def __init__(self):
        super().__init__()
        self._prepared_dvs = None

    def prepare_grouping(self, matrix: np.ndarray):
        matrix_flat = flatten_matrix(matrix)
        n_conn_max = np.max(np.sum(matrix_flat, axis=1))*2

        n_mat, n_src, n_tgt = matrix.shape[0], matrix.shape[1], matrix.shape[2]
        self._prepared_dvs = dv_loc_idx = np.zeros((n_mat, n_conn_max), dtype=int)
        idx_map = list(itertools.product(range(n_src), range(n_tgt)))
        for i_mat in range(n_mat):
            loc_idx = self._get_loc_indices(matrix_flat[i_mat, :], idx_map)
            dv_loc_idx[i_mat, :len(loc_idx)] = loc_idx

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

    def get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        if dv_idx >= self._prepared_dvs.shape[1]:
            return np.zeros((sub_matrix.shape[0],))
        return self._prepared_dvs[i_sub_matrix, dv_idx]


class RelCoordIndexLocationGrouper(CoordIndexLocationGrouper):
    """Group by relative connection location indices (encoded as i_src, i_tgt)"""

    @classmethod
    def _get_loc_indices(cls, conn_arr: np.ndarray, idx_map) -> np.ndarray:
        abs_loc_idx = super()._get_loc_indices(conn_arr, idx_map)
        rel_loc_idx = np.zeros(abs_loc_idx.shape)
        rel_loc_idx[0::2] = np.diff(np.concatenate([[0], abs_loc_idx[0::2]]))
        rel_loc_idx[1::2] = np.diff(np.concatenate([[0], abs_loc_idx[1::2]]))
        return rel_loc_idx
