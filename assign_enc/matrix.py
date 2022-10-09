import os
import math
import numba
import hashlib
import itertools
import numpy as np
from typing import *

__all__ = ['Node', 'AggregateAssignmentMatrixGenerator']


class Node:
    """
    Defines a source or target node with a specified number of possible outgoing/incoming connections:
    either an explicit list or only a lower bound.
    """

    def __init__(self, nr_conn_list: List[int] = None, min_conn: int = None, max_conn: int = math.inf,
                 repeated_allowed: bool = True, exists_conditionally=False):

        # Flag whether repeated connections to the same opposite node is allowed
        self.rep = repeated_allowed

        # Determine whether we have a set list of connections or only a lower bound
        conns, min_conns = None, None
        if nr_conn_list is not None:
            conns = nr_conn_list
            if exists_conditionally and 0 not in conns:
                conns = [0]+conns

        elif min_conn is not None:
            if exists_conditionally:
                min_conn = 0
            if max_conn == math.inf:
                min_conns = min_conn
            else:
                conns = list(range(min_conn, max_conn+1))
        else:
            raise ValueError('Either supply a list or a lower limit')
        self.conns = sorted(conns) if conns is not None else None
        self.min_conns = min_conns

    @property
    def max_inf(self):
        return self.conns is None

    def __str__(self):
        if self.conns is None:
            return f'({self.min_conns}..*)'
        return f'({",".join([str(i) for i in self.conns])})'

    def __repr__(self):
        return f'{self.__class__.__name__}(conns={self.conns!r}, min_conns={self.min_conns})'


class AggregateAssignmentMatrixGenerator:
    """
    Generator that iterates over all connection possibilities between two sets of objects, with various constraints:
    - Specified number or range of connections per object
    - Repeated connections between objects allowed or not
    - List of excluded connections

    If both the source and target lists contain at least one object which has no upper bound on the number of
    connections (meaning that in principle there are infinite connection possibilities), the upper bound is set to the
    number of connections enabled by the setting the source object connections to their minimum bounds (or 1 if the
    lower bound is 0).
    """

    def __init__(self, src: List[Node], tgt: List[Node], excluded: List[tuple] = None, max_conn: int = None):

        self.src = src
        self.tgt = tgt
        self._ex = None
        self.ex = excluded
        self._max_conn = max_conn
        self._min_src_appear = None
        self._max_src_appear = None
        self._max_tgt_appear = None

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
    def max_conn(self):
        if self._max_conn is None:
            self._max_conn = self._get_max_conn()
        return self._max_conn

    @property
    def min_src_appear(self):
        if self._min_src_appear is None:
            self._min_src_appear = self._get_min_appear(self.tgt)
        return self._min_src_appear

    @property
    def max_src_appear(self):
        if self._max_src_appear is None:
            self._max_src_appear = self._get_max_appear(self.tgt)
        return self._max_src_appear

    @property
    def max_tgt_appear(self):
        if self._max_tgt_appear is None:
            self._max_tgt_appear = self._get_max_appear(self.src)
        return self._max_tgt_appear

    def __iter__(self) -> Generator[np.array, None, None]:
        for src_nr in self._iter_sources():  # Iterate over source node nrs
            for tgt_nr in self._iter_targets(n_source=sum(src_nr)):  # Iterate over target node nrs
                yield from self._iter_matrices(src_nr, tgt_nr)  # Iterate over assignment matrices

    def get_agg_matrix(self, cache=False):
        cache_path = self._cache_path(f'{self._get_cache_key()}.npy')
        if cache and os.path.exists(cache_path):
            return np.load(cache_path)

        agg_matrix = self._agg_matrices(self)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, agg_matrix, allow_pickle=False)
        return agg_matrix

    def _get_cache_key(self):
        src_cache_key = ';'.join([repr(src) for src in self.src])
        tgt_cache_key = ';'.join([repr(tgt) for tgt in self.tgt])
        excluded_cache_key = ';'.join([f'{tup[0]:d},{tup[1]:d}' for tup in sorted([ex for ex in self._ex])]) \
            if self._ex is not None else ''
        cache_str = '||'.join([src_cache_key, tgt_cache_key, excluded_cache_key])
        return hashlib.md5(cache_str.encode('utf-8')).hexdigest()

    def _cache_path(self, sub_path=None):
        cache_folder = os.path.join(os.path.dirname(__file__), '.matrix_cache')
        return cache_folder if sub_path is None else os.path.join(cache_folder, sub_path)

    def _agg_matrices(self, matrix_gen):
        """Aggregate generated matrices into one big matrix"""
        matrices = list(matrix_gen)
        if len(matrices) == 0:
            return np.zeros((0, len(self.src), len(self.tgt)), dtype=int)

        agg_matrix = np.zeros((len(matrices), matrices[0].shape[0], matrices[0].shape[1]), dtype=int)
        for i, matrix in enumerate(matrices):
            agg_matrix[i, :, :] = matrix
        return agg_matrix

    def filter_matrices(self, matrix: np.ndarray, src_exists: List[bool] = None, tgt_exists: List[bool] = None) \
            -> np.ndarray:
        """Only keep matrices where non-existent nodes have no connections"""
        matrix_mask = np.ones((matrix.shape[0],), dtype=bool)
        if src_exists is None and tgt_exists is None:
            return matrix_mask

        # Deselect matrices where non-existing src nodes have one or more connections
        if src_exists is not None:
            for i, src_exists in enumerate(src_exists):
                if not src_exists:
                    matrix_mask &= np.sum(matrix[:, i, :], axis=1) == 0

        # Deselect matrices where non-existing tgt nodes have one or more connections
        if tgt_exists is not None:
            for i, tgt_exists in enumerate(tgt_exists):
                if not tgt_exists:
                    matrix_mask &= np.sum(matrix[:, :, i], axis=1) == 0

        return matrix_mask

    def iter_conns(self) -> Generator[List[Tuple[Node, Node]], None, None]:
        """Generate lists of edges from matrices"""
        for matrix in self:
            yield tuple(self.get_conns(matrix))

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

    def _iter_sources(self):
        yield from self._iter_conn_slots(self.src)

    def _iter_targets(self, n_source):
        yield from self._iter_conn_slots(self.tgt, is_src=False, n=n_source)

    def _iter_conn_slots(self, nodes: List[Node], is_src=True, n=None):
        """Iterate over all combinations of number of connections per nodes."""

        max_conn = self.max_src_appear if is_src else self.max_tgt_appear
        min_conn = self.min_src_appear if is_src else 0

        # Get all possible number of connections for each node
        n_conns = [self._get_conns(node, max_conn) for node in nodes]

        # Iterate over all combinations of number of connections
        for n_conn_nodes in itertools.product(*n_conns):

            n_tot = sum(n_conn_nodes)
            if n is not None and n_tot != n:
                continue
            if n_tot < min_conn:
                continue

            yield tuple(n_conn_nodes)

    @staticmethod
    def _get_conns(node: Node, max_conn: int):
        if node.max_inf:
            slot_max_conn = max_conn
            return list(range(node.min_conns, slot_max_conn + 1))
        return [c for c in node.conns if c <= max_conn]

    def _iter_matrices(self, n_src_conn, n_tgt_conn):
        """
        Iterate over all permutations of the connections between the source and target objects, taking repetition and
        exclusion constraints into account.

        Inspired by [Selva2017: Patterns in System Architecture Decisions], we are using the assigning pattern and
        representing assignments using a matrix of n_src x n_tgt. Each index (i_src, j_tgt) represents a connection
        between a specific source and target, and the value in the matrix represents the number of connections.

        To create the different matrices, we have to adhere to three constraints:
        1. The sum of target connections (sum over columns) should be equal to the nr of times they occur in tgt_objs
        2. Ensure that in certain rows (sources) or columns (targets), all values are max 1 if no repetition is allowed
        3. Block certain connections from being made at all if we have excluded edges
        """

        # Get list of source connection indices
        n_src_conn = np.array(n_src_conn, dtype=np.int64)
        n_src = len(n_src_conn)
        if n_src == 0:
            return

        # Prepare connection targets
        n_tgt = len(n_tgt_conn)
        n_tgt_conn = np.array(n_tgt_conn, dtype=np.int64)

        # Check which connections cannot be repeated
        no_repeat = np.zeros((n_src, n_tgt), dtype=bool)
        for i_src, src in enumerate(self.src):
            if not src.rep:
                no_repeat[i_src, :] = True
        for i_src, tgt in enumerate(self.tgt):
            if not tgt.rep:
                no_repeat[:, i_src] = True

        # Check which connections cannot be made at all
        ex = self.ex
        blocked = np.zeros((n_src, n_tgt), dtype=bool)
        for i_src, j_tgt in ex or []:
            blocked[i_src, j_tgt] = True
        not_blocked = ~blocked

        # Generate matrices
        yield from yield_matrices(n_src, n_tgt, n_src_conn, n_tgt_conn, no_repeat, not_blocked)

    def _get_max_conn(self) -> int:
        """
        Calculate the maximum number of connections that will be formed, taking infinite connections into account.
        """

        def _max(slots):
            n_conn = 0
            for slot in slots:
                n_conn += max(slot.conns) if not slot.max_inf else max([1, slot.min_conns])
            return n_conn

        return max([_max(self.src), _max(self.tgt)])

    @staticmethod
    def _get_min_appear(conn_tgt_nodes: List[Node]) -> int:
        n_appear = 0
        for tgt_node in conn_tgt_nodes:
            if tgt_node.max_inf:
                n_appear += tgt_node.min_conns
            else:
                n_appear += tgt_node.conns[0]
        return n_appear

    def _get_max_appear(self, conn_tgt_nodes: List[Node]) -> int:
        """
        Calculate the maximum number of times the connection nodes might appear considering repeated connection
        constraints.
        """
        n_max = self.max_conn
        n_appear = 0
        for tgt_node in conn_tgt_nodes:
            if tgt_node.rep:
                if tgt_node.max_inf:
                    return n_max
                n_appear += tgt_node.conns[-1]
            else:
                n_appear += 1
        return min(n_appear, n_max)


@numba.jit(nopython=True)
def yield_matrices(n_src: int, n_tgt: int, n_src_conn, n_tgt_conn, no_repeat, not_blocked):
    init_matrix = np.zeros((n_src, n_tgt), dtype=np.int64)
    init_tgt_sum = np.zeros((n_tgt,), dtype=np.int64)
    last_src_idx = n_src-1
    last_tgt_idx = n_tgt-1

    return _branch_matrices(0, init_matrix, init_tgt_sum, n_src_conn, n_tgt_conn, no_repeat, not_blocked, n_tgt,
                            last_src_idx, last_tgt_idx)


@numba.jit(nopython=True)
def _branch_matrices(i_src, current_matrix, current_sum, n_src_conn, n_tgt_conn, no_repeat, not_blocked, n_tgt,
                     last_src_idx, last_tgt_idx):
    # Get total connections from this source
    n_total_conns = n_src_conn[i_src]

    # Determine the amount of target connections available (per target node) for the current source node
    n_tgt_conn_avbl = n_tgt_conn-current_sum

    not_can_repeat = (no_repeat[i_src, :]) & (n_tgt_conn_avbl > 1)
    n_tgt_conn_avbl[not_can_repeat] = 1
    blocked = ~not_blocked[i_src, :]
    n_tgt_conn_avbl[blocked] = 0

    # Get number of minimum connections per target in order to be able to distribute all outgoing connections
    # starting from the first target
    reverse_cum_max_conn = np.cumsum(n_tgt_conn_avbl[::-1])[::-1]
    n_tgt_conn_min = (n_total_conns-reverse_cum_max_conn)[1:]

    # Loop over different source-to-targets connection patterns
    branched_matrices = []
    init_tgt_conns = np.zeros((n_tgt,), dtype=np.int64)
    for tgt_conns in _get_branch_tgt_conns(
            init_tgt_conns, 0, 0, n_total_conns, n_tgt_conn_avbl, n_tgt_conn_min, last_tgt_idx):
        # Modify matrix
        next_matrix = current_matrix.copy()
        next_matrix[i_src, :] = tgt_conns

        # Check if we can stop branching out
        if i_src == last_src_idx:
            branched_matrices.append(next_matrix)
            continue

        # Modify matrix sum
        next_sum = current_sum+tgt_conns

        # Branch out for next src node
        for branched_matrix in _branch_matrices(i_src+1, next_matrix, next_sum, n_src_conn, n_tgt_conn, no_repeat,
                                                not_blocked, n_tgt, last_src_idx, last_tgt_idx):
            branched_matrices.append(branched_matrix)

    return branched_matrices


@numba.jit(nopython=True)
def _get_branch_tgt_conns(tgt_conns, i_tgt, n_conn_set, n_total_conns, n_tgt_conn_avbl, n_tgt_conn_min, last_tgt_idx):
    # Check if we already have distributed all connections
    if n_conn_set == n_total_conns:
        return [tgt_conns]
    sub_tgt_conns = []

    # Branch: add another connection to the current target index, if below max for this target
    if tgt_conns[i_tgt] < n_tgt_conn_avbl[i_tgt]:
        next_tgt_conns = tgt_conns.copy()
        next_tgt_conns[i_tgt] += 1
        for sub_tgt_conn in _get_branch_tgt_conns(
                next_tgt_conns, i_tgt, n_conn_set+1, n_total_conns, n_tgt_conn_avbl, n_tgt_conn_min, last_tgt_idx):
            sub_tgt_conns.append(sub_tgt_conn)

    # Branch: move to next target, if we are not at the end and if we are above the minimum connections
    if i_tgt < last_tgt_idx and n_conn_set >= n_tgt_conn_min[i_tgt]:
        for sub_tgt_conn in _get_branch_tgt_conns(
                tgt_conns, i_tgt+1, n_conn_set, n_total_conns, n_tgt_conn_avbl, n_tgt_conn_min, last_tgt_idx):
            sub_tgt_conns.append(sub_tgt_conn)

    return sub_tgt_conns
