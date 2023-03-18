import functools
import numpy as np
from typing import *
from assign_enc.lazy_encoding import *
from assign_enc.lazy.encodings.on_demand_base import *
from assign_enc.lazy.encodings.conn_idx import ConnCombsEncoder
from assign_enc.eager.encodings.grouped_base import GroupedEncoder

__all__ = ['LazyAmountFirstEncoder', 'FlatLazyAmountEncoder', 'TotalLazyAmountEncoder', 'SourceLazyAmountEncoder',
           'SourceTargetLazyAmountEncoder', 'FlatLazyConnectionEncoder']


NList = List[Tuple[Tuple[int, ...], Tuple[int, ...]]]


class LazyAmountEncoder:
    """Base class for encoding a set of src and tgt connection amounts"""

    def _get_n_src_tgt_array(self, n_src_n_tgt: NList) -> np.ndarray:
        return np.array([n_src_conn+n_tgt_conn for n_src_conn, n_tgt_conn in n_src_n_tgt], dtype=int)

    def encode_prepare(self):
        pass

    def encode(self, n_src, n_tgt, n_src_n_tgt: NList, existence: NodeExistence, n_declared_start=None) \
            -> List[DiscreteDV]:
        raise NotImplementedError

    def decode(self, vector: DesignVector, existence: NodeExistence) \
            -> Optional[Tuple[DesignVector, Tuple[int, ...], Tuple[int, ...]]]:
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class LazyConnectionEncoder:
    """Base class for encoding connection selection"""

    def encode_prepare(self):
        pass

    def encode(self, n_src_n_tgt: NList, existence: NodeExistence, encoder: OnDemandLazyEncoder) -> List[DiscreteDV]:
        raise NotImplementedError

    def decode(self, n_src_conn, n_tgt_conn, vector: DesignVector, encoder: OnDemandLazyEncoder,
               existence: NodeExistence) -> Optional[Tuple[DesignVector, np.ndarray]]:
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class LazyAmountFirstEncoder(OnDemandLazyEncoder):
    """Base class for an encoder that first defines one or more design variables for selecting the amount of
    connections, and then selects the connection pattern. Generates matrices only when needed (on demand)."""

    def __init__(self, imputer: LazyImputer, amount_encoder: LazyAmountEncoder, conn_encoder: LazyConnectionEncoder):
        super().__init__(imputer)
        self.amount_encoder = amount_encoder
        self.conn_encoder = conn_encoder
        self._i_dv_amount = {}
        self._n_dv_amount_expand = {}
        self._n_dv_amount = {}
        self._i_dv_conn = {}
        self._n_dv_conn_expand = {}

    def _encode_prepare(self):
        super()._encode_prepare()

        self._i_dv_amount = {}
        self._n_dv_amount_expand = {}
        self._n_dv_amount = {}
        self._i_dv_conn = {}
        self._n_dv_conn_expand = {}

        self.amount_encoder.encode_prepare()
        self.conn_encoder.encode_prepare()

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        n_src_n_tgt = [(n_src_conn, n_tgt_conn)
                       for n_src_conn, n_tgt_conn, _ in self.iter_n_src_n_tgt(existence=existence)]
        dv_amount = self.amount_encoder.encode(self.n_src, self.tgt, n_src_n_tgt, existence)
        dv_amount, self._i_dv_amount[existence], self._n_dv_amount_expand[existence] = self._filter_dvs(dv_amount)
        self._n_dv_amount[existence] = len(dv_amount)

        dv_conn = self.conn_encoder.encode(n_src_n_tgt, existence, self)
        dv_conn, self._i_dv_conn[existence], self._n_dv_conn_expand[existence] = self._filter_dvs(dv_conn)

        return dv_amount+dv_conn

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[Tuple[DesignVector, np.ndarray]]:
        if existence not in self._n_dv_amount:
            return
        n_dv_amt = self._n_dv_amount[existence]

        amount_vector = vector[:n_dv_amt]
        amount_vector_expanded, _ = \
            self._unfilter_dvs(amount_vector, self._i_dv_amount[existence], self._n_dv_amount_expand[existence])

        n_src_tgt_conn = self.amount_encoder.decode(amount_vector_expanded, existence)
        if n_src_tgt_conn is None:
            return
        imp_amt_vector_expanded, n_src_conn, n_tgt_conn = n_src_tgt_conn
        imp_amount_vector_sel = np.array(imp_amt_vector_expanded)[self._i_dv_amount[existence]]
        imp_amount_vector = np.ones((len(amount_vector),), dtype=int)*X_INACTIVE_VALUE
        imp_amount_vector[:len(imp_amount_vector_sel)] = imp_amount_vector_sel

        conn_vector = vector[n_dv_amt:]
        conn_vector_expanded, _ = self._unfilter_dvs(
            conn_vector, self._i_dv_conn[existence], self._n_dv_conn_expand[existence])

        matrix_data = self.conn_encoder.decode(n_src_conn, n_tgt_conn, conn_vector_expanded, self, existence)
        if matrix_data is None:
            return
        imp_conn_vector_expanded, matrix = matrix_data
        imp_conn_vector_sel = np.array(imp_conn_vector_expanded)[self._i_dv_conn[existence]]
        imp_conn_vector = np.ones((len(conn_vector),), dtype=int)*X_INACTIVE_VALUE
        imp_conn_vector[:len(imp_conn_vector_sel)] = imp_conn_vector_sel

        imp_vector = list(imp_amount_vector)+list(imp_conn_vector)
        return imp_vector, matrix

    @staticmethod
    def group_by_values(values: np.ndarray, n_declared_start=None) -> np.ndarray:
        return GroupedEncoder.group_by_values(values, n_declared_start=n_declared_start)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r}, {self.amount_encoder!r}, {self.conn_encoder!r})'

    def __str__(self):
        return f'Lazy Amount First + {self._imputer!s}: {self.amount_encoder!s}; {self.conn_encoder!s}'


class FlatLazyAmountEncoder(LazyAmountEncoder):

    def __init__(self):
        self._n_src_n_tgt = {}

    def encode_prepare(self):
        self._n_src_n_tgt = {}

    def encode(self, n_src, n_tgt, n_src_n_tgt: NList, existence: NodeExistence, n_declared_start=None) \
            -> List[DiscreteDV]:
        self._n_src_n_tgt[existence] = n_src_n_tgt
        return [DiscreteDV(n_opts=len(n_src_n_tgt))]

    def decode(self, vector: DesignVector, existence: NodeExistence) \
            -> Optional[Tuple[DesignVector, Tuple[int, ...], Tuple[int, ...]]]:
        n_existence = self._n_src_n_tgt[existence]
        if len(n_existence) == 0:
            return

        idx = vector[0]
        if idx >= len(n_existence):
            idx = len(n_existence)-1
            vector = [idx]

        n_src, n_tgt = n_existence[idx]
        return vector, n_src, n_tgt

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'Flat Amount'


class GroupedLazyAmountEncoder(LazyAmountEncoder):

    def __init__(self):
        self._dv_val_map = {}

    def encode_prepare(self):
        self._dv_val_map = {}

    def encode(self, n_src, n_tgt, n_src_n_tgt: NList, existence: NodeExistence, n_declared_start=None) \
            -> List[DiscreteDV]:
        n_src_tgt_arr = self._get_n_src_tgt_array(n_src_n_tgt)
        if len(n_src_n_tgt) == 0:
            return []

        dv_group_values = self._get_dv_group_values(n_src, n_tgt, n_src_tgt_arr)
        dv_values = LazyAmountFirstEncoder.group_by_values(dv_group_values, n_declared_start=n_declared_start)
        self._dv_val_map[existence] = (ConnCombsEncoder.get_dv_map_for_lookup(dv_values), n_src_n_tgt)
        return [DiscreteDV(n_opts=n) for n in np.max(dv_values, axis=0)+1]

    def decode(self, vector: DesignVector, existence: NodeExistence) \
            -> Optional[Tuple[DesignVector, Tuple[int, ...], Tuple[int, ...]]]:

        if existence not in self._dv_val_map:
            return
        dv_map, n_src_n_tgt = self._dv_val_map[existence]
        idx_and_dv = ConnCombsEncoder.lookup_dv(dv_map, np.array(vector))
        if idx_and_dv is not None:
            i, dv = idx_and_dv
            n_src, n_tgt = n_src_n_tgt[i]
            return dv, n_src, n_tgt

    def _get_dv_group_values(self, n_src, n_tgt, n_src_tgt_arr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        raise NotImplementedError


class TotalLazyAmountEncoder(GroupedLazyAmountEncoder):

    def _get_dv_group_values(self, n_src, n_tgt, n_src_tgt_arr: np.ndarray) -> np.ndarray:
        return np.column_stack([
            np.sum(n_src_tgt_arr[:, :n_src], axis=1),
            np.arange(n_src_tgt_arr.shape[0]),
        ])

    def __str__(self):
        return 'Total Amount'


class SourceLazyAmountEncoder(GroupedLazyAmountEncoder):

    def _get_dv_group_values(self, n_src, n_tgt, n_src_tgt_arr: np.ndarray) -> np.ndarray:
        return np.column_stack([
            n_src_tgt_arr[:, :n_src],
            np.arange(n_src_tgt_arr.shape[0]),
        ])

    def __str__(self):
        return 'Source Amount'


class SourceTargetLazyAmountEncoder(GroupedLazyAmountEncoder):

    def _get_dv_group_values(self, n_src, n_tgt, n_src_tgt_arr: np.ndarray) -> np.ndarray:
        return n_src_tgt_arr

    def __str__(self):
        return 'Source Target Amount'


class FlatLazyConnectionEncoder(LazyConnectionEncoder):

    def __init__(self):
        self._n_exist_max = {}
        super().__init__()

    def encode_prepare(self):
        self._n_exist_max = {}

    def encode(self, n_src_n_tgt: NList, existence: NodeExistence, encoder: OnDemandLazyEncoder) -> List[DiscreteDV]:

        @functools.lru_cache(maxsize=None)
        def _count_matrices(n_src_, n_tgt_):
            return encoder.count_matrices(n_src_, n_tgt_)

        if len(n_src_n_tgt) == 0:
            return []
        n_conn = np.sum(np.array([n_src_conn for n_src_conn, _ in n_src_n_tgt]), axis=1)
        n_conn_max = np.max(n_conn)
        n_matrix_max = 0
        for i, n in enumerate(n_conn):
            if n == n_conn_max:
                n_matrix = _count_matrices(*n_src_n_tgt[i])
                if n_matrix > n_matrix_max:
                    n_matrix_max = n_matrix

        # Get the combination of src and tgt amounts that are closest to half of the max amounts
        # For half of the max amount (where max amount is given by the nr of nodes of the other type --> max src
        # connections is the amount of tgt nodes), there are the most amount of connection permutations possible
        n_src_tgt_half = np.array(list(encoder.matrix_gen.get_max_src_appear(existence=existence)) +
                                  list(encoder.matrix_gen.get_max_tgt_appear(existence=existence)))/2
        dist_to_half = np.empty((len(n_src_n_tgt),))
        for i, (n_src_conn, n_tgt_conn) in enumerate(n_src_n_tgt):
            dist_to_half[i] = np.sum(np.abs(np.array(list(n_src_conn)+list(n_tgt_conn))-n_src_tgt_half))

        min_dist_to_half = np.min(dist_to_half)
        for i in np.where(dist_to_half == min_dist_to_half)[0]:
            n_matrix = _count_matrices(*n_src_n_tgt[i])
            if n_matrix > n_matrix_max:
                n_matrix_max = n_matrix

        # Also check any src and tgt amounts where any of the two has all 1's for the non-zero entries
        is_all_ones = np.array([(np.max(n_src_conn) == 1 or np.max(n_tgt_conn) == 1) for n_src_conn, n_tgt_conn in n_src_n_tgt])
        if not np.all(is_all_ones):
            for i, is_all_one in enumerate(is_all_ones):
                if is_all_one:
                    n_matrix = _count_matrices(*n_src_n_tgt[i])
                    if n_matrix > n_matrix_max:
                        n_matrix_max = n_matrix

        # Also check any src and tgt amounts where all the amounts are the same
        for n_src_conn, n_tgt_conn in n_src_n_tgt:
            non_zero_src_conn = np.array(n_src_conn)
            non_zero_src_conn = non_zero_src_conn[non_zero_src_conn != 0]
            if len(non_zero_src_conn) == 0:
                continue

            non_zero_tgt_conn = np.array(n_tgt_conn)
            non_zero_tgt_conn = non_zero_tgt_conn[non_zero_tgt_conn != 0]
            if len(non_zero_tgt_conn) == 0:
                continue

            if np.all(non_zero_src_conn == non_zero_src_conn[0]) and np.all(non_zero_tgt_conn == non_zero_src_conn[0]):
                n_matrix = _count_matrices(n_src_conn, n_tgt_conn)
                if n_matrix > n_matrix_max:
                    n_matrix_max = n_matrix

        self._n_exist_max[existence] = n_matrix_max
        return [DiscreteDV(n_opts=n_matrix_max)] if n_matrix_max > 1 else []

    def decode(self, n_src_conn, n_tgt_conn, vector: DesignVector, encoder: OnDemandLazyEncoder,
               existence: NodeExistence) -> Optional[Tuple[DesignVector, np.ndarray]]:

        matrices = encoder.get_matrices(n_src_conn, n_tgt_conn)
        if matrices.shape[0] == 0:
            return vector, np.zeros((len(n_src_conn), len(n_tgt_conn)), dtype=int)
        if len(vector) == 0:
            return vector, matrices[0, :, :]

        if matrices.shape[0] > self._n_exist_max.get(existence, 0):
            raise RuntimeError(f'Sub-matrix found ({n_src_conn}, {n_tgt_conn}) with '
                               f'{matrices.shape[0]} > {self._n_exist_max.get(existence, 0)} possibilities')

        i_conn = vector[0]
        if i_conn >= matrices.shape[0]:
            i_conn = matrices.shape[0]-1
            vector = [i_conn]

        return vector, matrices[i_conn, :, :]

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'Flat Conn'
