import numpy as np
from typing import *
from assign_enc.lazy_encoding import *
from assign_enc.lazy.encodings.on_demand_base import *
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

    def encode(self, n_src, n_tgt, n_src_n_tgt: NList, existence: NodeExistence) -> List[DiscreteDV]:
        raise NotImplementedError

    def decode(self, vector: DesignVector, existence: NodeExistence) \
            -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
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
               existence: NodeExistence) -> Optional[np.ndarray]:
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

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[np.ndarray]:
        if existence not in self._n_dv_amount:
            return
        n_dv_amt = self._n_dv_amount[existence]

        amount_vector = vector[:n_dv_amt]
        amount_vector_expanded = self._unfilter_dvs(
            amount_vector, self._i_dv_amount[existence], self._n_dv_amount_expand[existence])
        if amount_vector_expanded is None:
            return
        n_src_tgt_conn = self.amount_encoder.decode(amount_vector_expanded, existence)
        if n_src_tgt_conn is None:
            return
        n_src_conn, n_tgt_conn = n_src_tgt_conn

        conn_vector = vector[n_dv_amt:]
        conn_vector_expanded = self._unfilter_dvs(
            conn_vector, self._i_dv_conn[existence], self._n_dv_conn_expand[existence])
        if conn_vector_expanded is None:
            return
        return self.conn_encoder.decode(n_src_conn, n_tgt_conn, conn_vector_expanded, self, existence)

    @staticmethod
    def group_by_values(values: np.ndarray) -> np.ndarray:
        return GroupedEncoder.group_by_values(values)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r}, {self.amount_encoder!r}, {self.conn_encoder!r})'

    def __str__(self):
        return f'Lazy Amount First + {self._imputer!s}: {self.amount_encoder!s}; {self.conn_encoder!s}'


class FlatLazyAmountEncoder(LazyAmountEncoder):

    def __init__(self):
        self._n_src_n_tgt = {}

    def encode_prepare(self):
        self._n_src_n_tgt = {}

    def encode(self, n_src, n_tgt, n_src_n_tgt: NList, existence: NodeExistence) -> List[DiscreteDV]:
        self._n_src_n_tgt[existence] = n_src_n_tgt
        return [DiscreteDV(n_opts=len(n_src_n_tgt))]

    def decode(self, vector: DesignVector, existence: NodeExistence) \
            -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        return self._n_src_n_tgt[existence][vector[0]]

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'Flat Amount'


class GroupedLazyAmountEncoder(LazyAmountEncoder):

    def __init__(self):
        self._dv_val_map = {}

    def encode_prepare(self):
        self._dv_val_map = {}

    def encode(self, n_src, n_tgt, n_src_n_tgt: NList, existence: NodeExistence) -> List[DiscreteDV]:
        n_src_tgt_arr = self._get_n_src_tgt_array(n_src_n_tgt)
        dv_group_values = self._get_dv_group_values(n_src, n_tgt, n_src_tgt_arr)
        dv_values = LazyAmountFirstEncoder.group_by_values(dv_group_values)
        self._dv_val_map[existence] = {tuple(dv_val): n_src_n_tgt[i] for i, dv_val in enumerate(dv_values)}
        return [DiscreteDV(n_opts=n) for n in np.max(dv_values, axis=0)+1]

    def decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        try:
            return self._dv_val_map[existence][tuple(vector)]
        except KeyError:
            pass

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

    def encode(self, n_src_n_tgt: NList, existence: NodeExistence, encoder: OnDemandLazyEncoder) -> List[DiscreteDV]:

        n_conn = np.sum(np.array([n_src_conn for n_src_conn, _ in n_src_n_tgt]), axis=1)
        n_conn_max = np.max(n_conn)
        n_matrix_max = 0
        for i, n in enumerate(n_conn):
            if n == n_conn_max:
                n_src_conn, n_tgt_conn = n_src_n_tgt[i]
                n_matrix = encoder.count_matrices(n_src_conn, n_tgt_conn)
                if n_matrix > n_matrix_max:
                    n_matrix_max = n_matrix

        return [DiscreteDV(n_opts=n_matrix_max)] if n_matrix_max > 1 else []

    def decode(self, n_src_conn, n_tgt_conn, vector: DesignVector, encoder: OnDemandLazyEncoder,
               existence: NodeExistence) -> Optional[np.ndarray]:

        matrices = encoder.get_matrices(n_src_conn, n_tgt_conn)
        if len(vector) == 0:
            if matrices.shape[0] == 0:
                return np.zeros((len(n_src_conn), len(n_tgt_conn)), dtype=int)
            return matrices[0, :, :]

        i_conn = vector[0]
        if i_conn < matrices.shape[0]:
            return matrices[i_conn, :, :]

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'Flat Conn'