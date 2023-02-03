import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.lazy_encoding import *
from assign_enc.eager.encodings.grouped_base import GroupedEncoder

__all__ = ['LazyConnIdxMatrixEncoder', 'ConnCombsEncoder', 'FlatConnCombsEncoder', 'GroupedConnCombsEncoder']


class ConnCombsEncoder:

    def __init__(self):
        self._registry = {}

    def encode_prepare(self):
        self._registry = {}

    def encode(self, key, matrix: np.ndarray) -> List[DiscreteDV]:
        raise NotImplementedError

    def decode(self, key, vector: DesignVector) -> Optional[np.ndarray]:
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class FlatConnCombsEncoder(ConnCombsEncoder):

    def encode(self, key, matrix: np.ndarray) -> List[DiscreteDV]:
        self._registry[key] = matrix
        return [DiscreteDV(n_opts=matrix.shape[0])] if matrix.shape[0] > 1 else []

    def decode(self, key, vector: DesignVector) -> Optional[np.ndarray]:
        matrix = self._registry[key]
        if matrix.shape[0] == 0:
            return np.zeros((matrix.shape[1],), dtype=np.int64)
        i_matrix = vector[0] if len(vector) > 0 else 0
        try:
            return matrix[i_matrix, :]
        except IndexError:
            return

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return 'Flat'


class GroupedConnCombsEncoder(ConnCombsEncoder):

    def encode(self, key, matrix: np.ndarray) -> List[DiscreteDV]:
        if matrix.shape[0] == 0:
            self._registry[key] = ({}, matrix)
            return []

        dv_values = GroupedEncoder.group_by_values(matrix)
        dv_map = {tuple(dv): i for i, dv in enumerate(dv_values)}
        self._registry[key] = (dv_map, matrix)
        return [DiscreteDV(n_opts=n+1) for n in np.max(dv_values, axis=0)]

    def decode(self, key, vector: DesignVector) -> Optional[np.ndarray]:
        dv_map, matrix = self._registry[key]
        if matrix.shape[0] == 0:
            return np.zeros((matrix.shape[1],), dtype=np.int64)
        if len(vector) == 0:
            return matrix[0, :]
        i_matrix = dv_map.get(tuple(vector))
        if i_matrix is None:
            return
        try:
            return matrix[i_matrix, :]
        except IndexError:
            return

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return 'Grouped'


class LazyConnIdxMatrixEncoder(LazyEncoder):
    """
    Encodes matrices by the location index of the connection, per source or target connector.
    """

    def __init__(self, imputer: LazyImputer, conn_encoder: ConnCombsEncoder, by_src=True, amount_first=False):
        self._conn_enc = conn_encoder
        self._by_src = by_src
        self._amount_first = amount_first
        super().__init__(imputer)

    def _encode_prepare(self):
        self._dv_idx_map = {}
        self._cache = {}
        self._conn_enc.encode_prepare()

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        matrix_gen = self._matrix_gen
        overall_max = matrix_gen.max_conn
        blocked_mask = matrix_gen.conn_blocked_mask
        no_repeat_mask = matrix_gen.no_repeat_mask

        if self._by_src:
            from_nodes, to_nodes = self.src, self.tgt
            has_from, has_to = existence.src_exists_mask(len(self.src)), existence.tgt_exists_mask(len(self.tgt))
            from_n_conn_override, to_n_conn_override = existence.src_n_conn_override, existence.tgt_n_conn_override
        else:
            from_nodes, to_nodes = self.tgt, self.src
            has_from, has_to = existence.tgt_exists_mask(len(self.tgt)), existence.src_exists_mask(len(self.src))
            from_n_conn_override, to_n_conn_override = existence.tgt_n_conn_override, existence.src_n_conn_override
            blocked_mask, no_repeat_mask = blocked_mask.T, no_repeat_mask.T

        self._dv_idx_map[existence] = dv_idx_map = {}
        all_dvs = []
        for i, node in enumerate(from_nodes):
            if not has_from[i]:
                continue
            n_from_conn = from_n_conn_override.get(i, matrix_gen.get_node_conns(node, overall_max))

            # Get maximum nr of connections for this source node
            n_tgt_conns = np.zeros((len(to_nodes),), dtype=np.int64)
            for j, to_node in enumerate(to_nodes):
                if not has_to[j] or blocked_mask[i, j]:
                    continue
                n_tgt_conns[j] = max(to_n_conn_override.get(j, matrix_gen.get_node_conns(to_node, overall_max)))
                if n_tgt_conns[j] > 1 and no_repeat_mask[i, j]:
                    n_tgt_conns[j] = 1

            # Get possible connection matrices
            all_failed = True
            matrix_combs_by_n = []
            for n_from in n_from_conn:
                cache_key = (n_from, tuple(n_tgt_conns))
                if cache_key in self._cache:
                    matrix_combs = self._cache[cache_key]
                else:
                    self._cache[cache_key] = matrix_combs = count_src_to_target(n_from, tuple(n_tgt_conns))

                if matrix_combs.shape[0] > 0:
                    all_failed = False

                matrix_combs_by_n.append(matrix_combs)

            # If for any of the connection sources there is no way to make any connection, this is true for all
            if all_failed:
                return []

            # Encode
            keys_n_dv = []
            dvs = []
            if self._amount_first:
                if len(matrix_combs_by_n) > 1:
                    dvs.append(DiscreteDV(n_opts=len(matrix_combs_by_n)))
                conn_dvs = []
                for k, matrix in enumerate(matrix_combs_by_n):
                    key = (existence, i, k)
                    conn_dv = self._conn_enc.encode(key, matrix)
                    conn_dvs.append(conn_dv)
                    keys_n_dv.append((key, len(conn_dv)))
                dvs += LazyEncoder._merge_design_vars(conn_dvs)

            else:
                key = (existence, i)
                conn_dv = self._conn_enc.encode(key, np.row_stack(matrix_combs_by_n))
                keys_n_dv.append((key, len(conn_dv)))
                dvs += conn_dv

            dv_idx_map[i] = (len(all_dvs), len(dvs), keys_n_dv)
            all_dvs += dvs
        return all_dvs

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[np.ndarray]:
        by_src, amount_first = self._by_src, self._amount_first
        dv_idx_map = self._dv_idx_map.get(existence, {})
        matrix = np.zeros((self.n_src, self.n_tgt), dtype=int)
        if not by_src:
            matrix = matrix.T

        for i, (i_dv_start, n_dv, keys) in dv_idx_map.items():
            vector_i = vector[i_dv_start:i_dv_start+n_dv]
            i_key = 0
            if amount_first:
                if len(keys) > 1:
                    i_key = vector_i[0]
                    vector_i = vector_i[1:]
            try:
                key, n_dv_i = keys[i_key]
            except IndexError:
                return
            sub_matrix = self._conn_enc.decode(key, vector_i[:n_dv_i])
            if sub_matrix is None:
                return
            matrix[i, :] = sub_matrix

        return matrix if by_src else matrix.T

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r}, {self._conn_enc!r}, ' \
               f'by_src={self._by_src}, amount_first={self._amount_first})'

    def __str__(self):
        return f'Lazy Conn Idx By {"Src" if self._by_src else "Tgt"}{" Amnt" if self._amount_first else ""} + ' \
               f'{self._conn_enc!s} + {self._imputer!s}'
