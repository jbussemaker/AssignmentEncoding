import os
import math
import numba
import pickle
import hashlib
import itertools
import functools
import numpy as np
from typing import *
from dataclasses import dataclass
from assign_enc.cache import get_cache_path

__all__ = ['Node', 'NodeExistence', 'NodeExistencePatterns', 'AggregateAssignmentMatrixGenerator', 'count_n_pool_take',
           'MatrixMap', 'MatrixMapOptional', 'count_src_to_target']


class Node:
    """
    Defines a source or target node with a specified number of possible outgoing/incoming connections:
    either an explicit list or only a lower bound.
    """

    def __init__(self, nr_conn_list: List[int] = None, min_conn: int = None, max_conn: int = math.inf,
                 repeated_allowed: bool = True):

        # Flag whether repeated connections to the same opposite node is allowed
        self.rep = repeated_allowed

        # Determine whether we have a set list of connections or only a lower bound
        conns, min_conns = None, None
        if nr_conn_list is not None:
            conns = nr_conn_list

        elif min_conn is not None:
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
        return f'{self.__class__.__name__}(conns={self.conns!r}, min_conns={self.min_conns}, rep={self.rep}'


class NodeExistence:
    """
    Defines a specific src/tgt node existence pattern.
    """

    def __init__(
            self,
            src_exists: Union[List[bool], np.ndarray] = None,
            tgt_exists: Union[List[bool], np.ndarray] = None,
            src_n_conn_override: Dict[int, List[int]] = None,
            tgt_n_conn_override: Dict[int, List[int]] = None,
    ):
        self.src_n_conn_override = self._get_n_conn_override(src_exists, src_n_conn_override)
        self.tgt_n_conn_override = self._get_n_conn_override(tgt_exists, tgt_n_conn_override)
        self._hash = None

    @staticmethod
    def _get_n_conn_override(exists: Union[List[bool], np.ndarray] = None,
                             n_conn_override: Dict[int, List[int]] = None) -> Dict[int, List[int]]:
        if n_conn_override is None and exists is None:
            return {}

        n_conn_override = {} if n_conn_override is None else n_conn_override.copy()
        if exists is not None:
            for i, exist in enumerate(exists):
                if not exist:
                    n_conn_override[i] = [0]
        if len(n_conn_override) == 0:
            return {}
        return n_conn_override

    def has_src(self, i):
        return self.src_n_conn_override.get(i) != [0]

    def has_tgt(self, i):
        return self.tgt_n_conn_override.get(i) != [0]

    def src_exists_mask(self, n_src: int):
        exist_mask = np.ones((n_src,), dtype=bool)
        for i in range(n_src):
            if not self.has_src(i):
                exist_mask[i] = False
        return exist_mask

    def tgt_exists_mask(self, n_tgt: int):
        exist_mask = np.ones((n_tgt,), dtype=bool)
        for i in range(n_tgt):
            if not self.has_tgt(i):
                exist_mask[i] = False
        return exist_mask

    def none_exists(self, n_src: int, n_tgt: int):
        if np.any(self.src_exists_mask(n_src)):
            return False
        if np.any(self.tgt_exists_mask(n_tgt)):
            return False
        return True

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((
                tuple([(k, tuple(v)) for k, v in self.src_n_conn_override.items()]) if self.src_n_conn_override is not None else -1,
                tuple([(k, tuple(v)) for k, v in self.tgt_n_conn_override.items()]) if self.tgt_n_conn_override is not None else -1,
            ))
        return self._hash

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        src_conn_str = '' if self.src_n_conn_override is None else ';'.join([f'{k}:{",".join([str(vv) for vv in v])}' for k, v in self.src_n_conn_override.items()])
        tgt_conn_str = '' if self.tgt_n_conn_override is None else ';'.join([f'{k}:{",".join([str(vv) for vv in v])}' for k, v in self.tgt_n_conn_override.items()])
        return f'{src_conn_str} _ {tgt_conn_str}'


@dataclass(frozen=False)
class NodeExistencePatterns:
    patterns: List[NodeExistence]

    def __post_init__(self):
        if len(set(self.patterns)) != len(self.patterns):
            raise RuntimeError('Duplicate node existence patterns')

    @classmethod
    def always_exists(cls) -> 'NodeExistencePatterns':
        return cls([NodeExistence()])

    @classmethod
    def get_all_combinations(cls, src_is_conditional: List[bool], tgt_is_conditional: List[bool]):
        """
        Gets patterns for all combination of conditional existence, e.g.:
        is_conditional = [False, True, True]
        exists = [True, False, False]
        exists = [True, True , False]
        exists = [True, False, True ]
        exists = [True, True , True ]
        """
        src_exist_opts = [[True, False] if is_conditional else [True] for is_conditional in src_is_conditional]
        tgt_exist_opts = [[True, False] if is_conditional else [True] for is_conditional in tgt_is_conditional]

        patterns = []
        for src_exists in itertools.product(*src_exist_opts):
            for tgt_exists in itertools.product(*tgt_exist_opts):
                patterns.append(NodeExistence(src_exists=list(src_exists), tgt_exists=list(tgt_exists)))
        return cls(patterns)

    @classmethod
    def get_increasing(cls, src_is_conditional: List[bool], tgt_is_conditional: List[bool]):
        """
        Gets patterns for conditional existence combinations where node existence is monotonically increasing, e.g.:
        is_conditional = [False, True, True]
        exists = [True, False, False]
        exists = [True, True , False]
        exists = [True, True , True ]
        """

        def _get_exist_opts(is_conditional: List[bool]) -> List[List[bool]]:
            n_cond = sum([is_cond for is_cond in is_conditional])
            exist_opts = []
            for n_cond_exists in range(n_cond+1):
                n_exists = 0
                exist_opt = []
                for is_cond in is_conditional:
                    if is_cond:
                        exist_opt.append(n_exists < n_cond_exists)
                        n_exists += 1
                    else:
                        exist_opt.append(True)
                exist_opts.append(exist_opt)
            return exist_opts

        src_exist_opts = _get_exist_opts(src_is_conditional)
        tgt_exist_opts = _get_exist_opts(tgt_is_conditional)
        return cls([NodeExistence(src_exists=list(src_exists), tgt_exists=list(tgt_exists))
                    for src_exists, tgt_exists in itertools.product(src_exist_opts, tgt_exist_opts)])

    def src_is_conditional(self, i):
        for pattern in self.patterns:
            if not pattern.has_src(i):
                return True
        return False

    def tgt_is_conditional(self, i):
        for pattern in self.patterns:
            if not pattern.has_tgt(i):
                return True
        return False


MatrixMap = Dict[NodeExistence, np.ndarray]
MatrixMapOptional = Union[np.ndarray, Dict[NodeExistence, np.ndarray]]


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

    def __init__(self, src: List[Node], tgt: List[Node], excluded: List[tuple] = None,
                 existence_patterns: NodeExistencePatterns = None, max_conn: int = None):

        self.src = src
        self.tgt = tgt
        self._ex = None
        self.ex = excluded
        self._exist = None
        self.existence_patterns = existence_patterns
        self._max_conn = max_conn
        self._max_src_appear = None
        self._max_tgt_appear = None
        self._no_repeat_mat = None
        self._conn_blocked_mat = None

    @property
    def ex(self) -> Optional[Set[Tuple[int, int]]]:
        return self._ex

    @ex.setter
    def ex(self, excluded: Optional[List[Tuple[Node, Node]]]):
        if excluded is None:
            self._ex = None
        else:
            self._ex = self.ex_to_idx(self.src, self.tgt, excluded)
        self._conn_blocked_mat = None

    @staticmethod
    def ex_to_idx(src: List[Node], tgt: List[Node], excluded: Optional[List[Tuple[Node, Node]]]) \
            -> Optional[Set[Tuple[int, int]]]:
        if excluded is None:
            return excluded
        src_idx = {s: i for i, s in enumerate(src)}
        tgt_idx = {t: i for i, t in enumerate(tgt)}
        excluded_idx = set([(src_idx[ex[0]], tgt_idx[ex[1]]) for ex in excluded])
        return excluded_idx if len(excluded_idx) > 0 else None

    @property
    def existence_patterns(self) -> NodeExistencePatterns:
        return self._exist

    @existence_patterns.setter
    def existence_patterns(self, existence_patterns: NodeExistencePatterns = None):
        if existence_patterns is None:
            existence_patterns = NodeExistencePatterns.always_exists()
        self._exist = existence_patterns

    @property
    def max_conn(self):
        if self._max_conn is None:
            self._max_conn = self._get_max_conn()
        return self._max_conn

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

    @property
    def no_repeat_mask(self):
        """Matrix mask that specifies whether for the connections repetition is not allowed (True value)"""
        if self._no_repeat_mat is None:
            self._no_repeat_mat = no_repeat = np.zeros((len(self.src), len(self.tgt)), dtype=bool)
            for i_src, src in enumerate(self.src):
                if not src.rep:
                    no_repeat[i_src, :] = True
            for i_src, tgt in enumerate(self.tgt):
                if not tgt.rep:
                    no_repeat[:, i_src] = True
        return self._no_repeat_mat

    @property
    def conn_blocked_mask(self):
        """Mask that specifies whether connections are blocked (True) or not (False)"""
        if self._conn_blocked_mat is None:
            ex = self.ex
            self._conn_blocked_mat = blocked = np.zeros((len(self.src), len(self.tgt)), dtype=bool)
            for i_src, j_tgt in ex or []:
                blocked[i_src, j_tgt] = True
        return self._conn_blocked_mat

    def __iter__(self) -> Generator[Tuple[np.array, NodeExistence], None, None]:
        for n_src_conn, n_tgt_conn, existence in self.iter_n_sources_targets():
            yield self._iter_matrices(n_src_conn, n_tgt_conn), existence  # Iterate over assignment matrices

    def iter_matrices(self, existence: NodeExistence = None):
        for n_src_conn, n_tgt_conn, exist in self.iter_n_sources_targets(existence=existence):
            for matrix in self._iter_matrices(n_src_conn, n_tgt_conn):
                yield matrix, exist

    def get_agg_matrix(self, cache=False) -> MatrixMap:
        if cache:
            agg_matrix = self._load_agg_matrix_from_cache()
            if agg_matrix is not None:
                return agg_matrix

        agg_matrix = self._agg_matrices(self, ensure_all_existence=True)
        self._write_to_cache(self._get_cache_file(), agg_matrix)
        return agg_matrix

    def _load_agg_matrix_from_cache(self) -> Optional[MatrixMap]:
        return self._load_from_cache(self._get_cache_file())

    def _write_to_cache(self, cache_path, obj):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as fp:
            pickle.dump(obj, fp)

    def _load_from_cache(self, cache_path):
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as fp:
                return pickle.load(fp)

    def reset_agg_matrix_cache(self):
        cache_path = self._get_cache_file()
        if os.path.exists(cache_path):
            os.remove(cache_path)

        cache_path = self._get_cache_file_iter_n()
        if os.path.exists(cache_path):
            os.remove(cache_path)

    def _get_cache_file(self):
        return self._cache_path(f'{self._get_cache_key()}.pkl')

    def _get_cache_file_iter_n(self):
        return self._cache_path(f'{self._get_cache_key()}_iter_n.pkl')

    def _get_cache_key(self):
        return self.get_cache_key(self.src, self.tgt, self._ex, self.existence_patterns)

    @staticmethod
    def get_cache_key(src: List[Node], tgt: List[Node], excluded, existence_patterns):
        src_cache_key = ';'.join([repr(s) for s in src])
        tgt_cache_key = ';'.join([repr(t) for t in tgt])
        excluded_cache_key = ';'.join([f'{tup[0]:d},{tup[1]:d}' for tup in sorted([ex for ex in excluded])]) \
            if excluded is not None else ''
        exist_cache_key = ';'.join([str(hash(p)) for p in existence_patterns.patterns])\
            if existence_patterns is not None else ''
        cache_str = '||'.join([src_cache_key, tgt_cache_key, excluded_cache_key, exist_cache_key])
        return hashlib.md5(cache_str.encode('utf-8')).hexdigest()

    def _cache_path(self, sub_path=None):
        sel_cache_folder = 'matrix_cache'
        sub_path = os.path.join(sel_cache_folder, sub_path) if sub_path is not None else sel_cache_folder
        return get_cache_path(sub_path=sub_path)

    def _agg_matrices(self, matrix_gen, ensure_all_existence=False):
        """Aggregate generated matrices into one big matrix, one per existence pattern"""
        matrices_by_existence = {}
        for matrix_data in list(matrix_gen):
            if isinstance(matrix_data, np.ndarray):
                matrices, existence = matrix_data, NodeExistence()
            else:
                matrices, existence = matrix_data

            if existence not in matrices_by_existence:
                matrices_by_existence[existence] = []
            matrices_by_existence[existence].append(matrices)

        if ensure_all_existence:
            for existence in self.iter_existence():
                if existence not in matrices_by_existence:
                    matrices_by_existence[existence] = []

        agg_matrices_by_existence = {}
        for existence, matrices in matrices_by_existence.items():
            if len(matrices) == 0:
                agg_matrices_by_existence[existence] = np.zeros((0, len(self.src), len(self.tgt)), dtype=int)
            elif len(matrices) == 1:
                agg_matrices_by_existence[existence] = matrices[0]
            else:
                agg_matrices_by_existence[existence] = np.row_stack(matrices)
        return agg_matrices_by_existence

    def get_matrix_for_existence(self, matrix: MatrixMapOptional, existence: NodeExistence = None) -> np.ndarray:
        if isinstance(matrix, np.ndarray):
            return matrix
        if existence not in matrix:
            return np.empty((0, len(self.src), len(self.tgt)), dtype=int)
        return matrix[existence]

    def iter_conns(self) -> Generator[List[Tuple[Node, Node]], None, None]:
        """Generate lists of edges from matrices"""
        for matrix, _ in self.iter_matrices():
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

    def iter_n_sources_targets(self, existence: NodeExistence = None, cache=True):
        cache_path = self._get_cache_file_iter_n()
        if cache:
            iter_data = self._load_from_cache(cache_path)
            if iter_data is not None:
                n_src = len(self.src)
                for exist, np_tup in iter_data:
                    if existence is None or existence == exist:
                        for i in range(np_tup.shape[0]):
                            yield tuple(np_tup[i, :n_src]), tuple(np_tup[i, n_src:]), exist
                return

        tuples = {}
        for n_src_conn, n_tgt_conn, exist in self._iter_n_sources_targets():
            if exist not in tuples:
                tuples[exist] = []
            tuples[exist].append((n_src_conn, n_tgt_conn))
            if existence is None or existence == exist:
                yield n_src_conn, n_tgt_conn, exist

        # Store it as a numpy array (more efficient than storing as tuples)
        tuples_efficient = []
        n_src = len(self.src)
        for exist, n_src_n_tgt in tuples.items():
            np_tup = np.empty((len(n_src_n_tgt), n_src+len(self.tgt)), dtype=np.int64)
            for i, (n_src_conn, n_tgt_conn) in enumerate(n_src_n_tgt):
                np_tup[i, :n_src] = n_src_conn
                np_tup[i, n_src:] = n_tgt_conn
            tuples_efficient.append((exist, np_tup))
        self._write_to_cache(cache_path, tuples_efficient)

    def _iter_n_sources_targets(self, existence: NodeExistence = None):
        for exist in (self.iter_existence() if existence is None else [existence]):
            for n_src_conn in self.iter_sources(existence=exist):
                for n_tgt_conn in self.iter_targets(n_source=sum(n_src_conn), existence=exist):
                    yield n_src_conn, n_tgt_conn, exist

    def iter_existence(self):
        yield from self.existence_patterns.patterns

    def iter_sources(self, existence: NodeExistence = None):
        n_conn_override = existence.src_n_conn_override if existence is not None else None
        yield from self._iter_conn_slots(self.src, n_conn_override=n_conn_override)

    def iter_targets(self, n_source, existence: NodeExistence = None):
        n_conn_override = existence.tgt_n_conn_override if existence is not None else None
        yield from self._iter_conn_slots(self.tgt, is_src=False, n=n_source, n_conn_override=n_conn_override)

    def _iter_conn_slots(self, nodes: List[Node], is_src=True, n=None, n_conn_override: Dict[int, List[int]] = None):
        """Iterate over all combinations of number of connections per nodes."""

        max_conn = self.max_src_appear if is_src else self.max_tgt_appear

        if n_conn_override is None:
            n_conn_override = {}

        # Get all possible number of connections for each node
        n_conns = []
        for i, node in enumerate(nodes):
            if i in n_conn_override:
                n_conns.append(n_conn_override[i])
            else:
                n_conns.append(self.get_node_conns(node, max_conn))

        # If we have an amount constraint, use the matrix generation algorithm
        if n is not None:
            n_conn_max = [max(n_conn) for n_conn in n_conns]
            n_tgt_combs = count_src_to_target(n_src=n, n_src_to_target=tuple(n_conn_max))

            # Filter out invalid amounts
            for i_tgt, n_conn in enumerate(n_conns):
                tgt_n_conns = n_tgt_combs[:, i_tgt]
                n_invalid = set(tgt_n_conns).difference(set(n_conn))
                if len(n_invalid) > 0:
                    invalid_mask = np.zeros((len(tgt_n_conns),), dtype=bool)
                    for n in n_invalid:
                        invalid_mask |= tgt_n_conns == n
                    n_tgt_combs = n_tgt_combs[~invalid_mask, :]

            for i in reversed(range(n_tgt_combs.shape[0])):  # Reversed for compatibility
                yield tuple(n_tgt_combs[i, :])
            return

        # Iterate over all combinations of number of connections
        for n_conn_nodes in itertools.product(*n_conns):

            n_tot = sum(n_conn_nodes)
            if n is not None and n_tot != n:
                continue

            yield tuple(n_conn_nodes)

    def validate_matrix(self, matrix: np.ndarray, existence: NodeExistence = None) -> bool:
        """Checks whether a connection matrix is valid"""

        # Check if any connections have repetitions that are not allowed
        if np.any((matrix > 1) & self.no_repeat_mask):
            return False

        # Check if any connections are made that are not allowed
        if np.any((matrix > 0) & self.conn_blocked_mask):
            return False

        # Check number of source connections
        if existence is None:
            existence = NodeExistence()
        src_n_conn_override = existence.src_n_conn_override if existence.src_n_conn_override is not None else {}
        max_conn = self.max_src_appear
        for i, n_src in enumerate(np.sum(matrix, axis=1)):
            if not existence.has_src(i):
                if n_src > 0:
                    return False
                else:
                    continue
            if i in src_n_conn_override and n_src not in src_n_conn_override[i]:
                return False
            if not self._check_conns(n_src, self.src[i], max_conn):
                return False

        # Check number of target connections
        max_conn = self.max_tgt_appear
        tgt_n_conn_override = existence.tgt_n_conn_override if existence.tgt_n_conn_override is not None else {}
        for i, n_tgt in enumerate(np.sum(matrix, axis=0)):
            if not existence.has_tgt(i):
                if n_tgt > 0:
                    return False
                else:
                    continue
            if i in tgt_n_conn_override and n_tgt not in tgt_n_conn_override[i]:
                return False
            if not self._check_conns(n_tgt, self.tgt[i], max_conn):
                return False

        return True

    @staticmethod
    def _check_conns(n_conns: int, node: Node, max_conn: int) -> bool:
        if n_conns > max_conn:
            return False
        if node.max_inf:
            return node.min_conns <= n_conns
        return n_conns in node.conns

    @staticmethod
    def get_node_conns(node: Node, max_conn: int):
        if node.max_inf:
            slot_max_conn = max_conn
            return list(range(node.min_conns, slot_max_conn+1))

        return [c for c in node.conns if c <= max_conn]

    def count_all_matrices(self, max_by_existence=True) -> int:
        agg_matrix = self._load_agg_matrix_from_cache()
        if agg_matrix is not None:
            count_by_existence = {existence: matrix.shape[0] for existence, matrix in agg_matrix.items()}
        else:
            count_by_existence = {}
            for n_src_conn, n_tgt_conn, existence in self.iter_n_sources_targets():
                if existence not in count_by_existence:
                    count_by_existence[existence] = 0
                count_by_existence[existence] += self.count_matrices(n_src_conn, n_tgt_conn)

        if max_by_existence:
            return max(count_by_existence.values())
        return sum(count_by_existence.values())

    def get_matrices_by_n_conn(self, n_src_conn, n_tgt_conn):
        return self._iter_matrices(n_src_conn, n_tgt_conn)

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

        # Check if we have any source connections
        if len(n_src_conn) == 0:
            return np.empty((0, 0, len(n_tgt_conn)), dtype=int)

        # Generate matrices
        return self._get_matrices_numpy(n_src_conn, n_tgt_conn)  # No compilation time
        # return self._get_matrices_numba(n_src_conn, n_tgt_conn)  # Slightly faster

    def count_matrices(self, n_src_conn, n_tgt_conn) -> int:
        """Count the number of permutations as would be generated by _iter_matrices."""

        # Count special cases
        n_mat = self._count_matrices_special(n_src_conn, n_tgt_conn)
        if n_mat is not None:
            return n_mat

        # Count symmetric special cases
        n_mat = self._count_matrices_special(n_tgt_conn, n_src_conn, is_switched=True)
        if n_mat is not None:
            return n_mat

        return self._get_matrices_numpy(n_src_conn, n_tgt_conn, create_matrices=False)  # No compilation time
        # return len(self._get_matrices_numba(n_src_conn, n_tgt_conn, output_matrix=False))  # Slightly faster

    def _get_matrices_numba(self, n_src_conn, n_tgt_conn, output_matrix=True):
        n_src_conn = np.array(n_src_conn, dtype=np.int64)
        n_tgt_conn = np.array(n_tgt_conn, dtype=np.int64)
        matrices = yield_matrices(
            len(n_src_conn), len(n_tgt_conn), np.array(n_src_conn, dtype=np.int64),
            np.array(n_tgt_conn, dtype=np.int64), self.no_repeat_mask, self.conn_blocked_mask)

        if output_matrix:
            if len(matrices) == 0:
                return np.empty((0, len(n_src_conn), len(n_tgt_conn)), dtype=int)
            elif len(matrices) == 1:
                return np.array([matrices[0]])
            return np.array(matrices)
        return matrices

    def _count_matrices_special(self, n_src_conn, n_tgt_conn, is_switched=False) -> Optional[int]:
        # No connections
        n_src_total = sum(n_src_conn)
        n_tgt_total = sum(n_tgt_conn)
        if n_src_total == 0 or n_tgt_total == 0:
            return 1

        # 1 outgoing connection
        if len(n_src_conn) == 1 and n_src_conn[0] == 1:
            n_mat = 0
            for n_tgt in n_tgt_conn:
                if n_tgt > 0:
                    n_mat += 1
            return n_mat

        # Get max conn
        max_conn = np.ones((len(n_src_conn), len(n_tgt_conn)), dtype=int)*max(n_src_total, n_tgt_total)
        if is_switched:
            max_conn = max_conn.T
        max_conn[self.no_repeat_mask] = 1
        max_conn[self.conn_blocked_mask] = 0
        if is_switched:
            max_conn = max_conn.T

        # One of the src nodes has same nr of target connections
        for i_src, n_src in enumerate(n_src_conn):
            if n_src == n_tgt_total:
                # Check if any of the connection nr constraints are violated
                if np.any(n_tgt_conn > max_conn[i_src, :]):
                    return 0
                return 1

    def _get_matrices_numpy(self, n_src_conn, n_tgt_conn, create_matrices=True):
        no_repeat_mask = self.no_repeat_mask
        conn_blocked_mask = self.conn_blocked_mask
        return _get_matrices(
            tuple(n_src_conn), tuple(n_tgt_conn), tuple(no_repeat_mask.ravel()), tuple(conn_blocked_mask.ravel()),
            create_matrices=create_matrices)

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
    def _get_min_conns(node: Node) -> int:
        if node.max_inf:
            return node.min_conns
        return node.conns[0]

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


@numba.njit()
def yield_matrices(n_src: int, n_tgt: int, n_src_conn, n_tgt_conn, no_repeat, blocked):
    init_matrix = np.zeros((n_src, n_tgt), dtype=np.int64)
    init_tgt_sum = np.zeros((n_tgt,), dtype=np.int64)
    last_src_idx = n_src-1
    last_tgt_idx = n_tgt-1

    return _branch_matrices(0, init_matrix, init_tgt_sum, n_src_conn, n_tgt_conn, no_repeat, blocked, n_tgt,
                            last_src_idx, last_tgt_idx)


@numba.njit()
def _branch_matrices(i_src, current_matrix, current_sum, n_src_conn, n_tgt_conn, no_repeat, blocked, n_tgt,
                     last_src_idx, last_tgt_idx):
    # Get total connections from this source
    n_total_conns = n_src_conn[i_src]

    # Determine the amount of target connections available (per target node) for the current source node
    n_tgt_conn_avbl = n_tgt_conn-current_sum

    not_can_repeat = (no_repeat[i_src, :]) & (n_tgt_conn_avbl > 1)
    n_tgt_conn_avbl[not_can_repeat] = 1
    n_tgt_conn_avbl[blocked[i_src, :]] = 0

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
                                                blocked, n_tgt, last_src_idx, last_tgt_idx):
            branched_matrices.append(branched_matrix)

    return branched_matrices


@numba.njit()
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


@functools.lru_cache(maxsize=None)
def _get_matrices(n_src_conn, n_tgt_conn, no_repeat_mask_tup, conn_blocked_mask_tup, create_matrices=True):
    """Recursive column-wise (target-wise)"""
    n_src, n_tgt = len(n_src_conn), len(n_tgt_conn)
    no_repeat_mask = np.array(no_repeat_mask_tup).reshape(n_src, n_tgt)
    conn_blocked_mask = np.array(conn_blocked_mask_tup).reshape(n_src, n_tgt)

    # Determine number of src connections to first target
    n_target_from_src = np.array(n_src_conn)
    n_target_from_src[no_repeat_mask[:, 0] & (n_target_from_src > 0)] = 1
    n_target_from_src[conn_blocked_mask[:, 0]] = 0

    # Get combinations of sources to target assignments
    create_here = create_matrices or len(n_tgt_conn) > 1
    n_taken_combs = count_src_to_target(
        n_tgt_conn[0], tuple(n_target_from_src), create_matrices=create_here)
    if (n_taken_combs.shape[0] if create_here else n_taken_combs) == 0:
        return np.empty((0, n_src, n_tgt), dtype=int) if create_matrices else 0

    # Only one target left to connect to
    if len(n_tgt_conn) == 1:
        if create_matrices:
            return n_taken_combs.reshape((n_taken_combs.shape[0], n_src, 1))
        return n_taken_combs

    # Get combinations for first column
    matrices = []
    count = 0
    n_src_conn_arr = np.array(n_src_conn)
    for i_comb, n_from_src in enumerate(n_taken_combs):
        n_src_conn_remain = n_src_conn_arr-n_from_src
        next_nr_mask = tuple(no_repeat_mask[:, 1:].ravel())
        next_cb_mask = tuple(conn_blocked_mask[:, 1:].ravel())

        # Get combinations for further columns
        next_matrices = _get_matrices(
            tuple(n_src_conn_remain), n_tgt_conn[1:], next_nr_mask, next_cb_mask, create_matrices=create_matrices)

        if not create_matrices:
            count += next_matrices
            continue

        matrices_i = np.empty((next_matrices.shape[0], n_src, n_tgt), dtype=int)
        matrices_i[:, :, 0] = n_from_src
        matrices_i[:, :, 1:] = next_matrices
        matrices.append(matrices_i)

    if not create_matrices:
        return count
    return matrices[0] if len(matrices) == 1 else np.row_stack(matrices)


@functools.lru_cache(maxsize=None)
# @numba.njit()
def count_src_to_target(n_src, n_src_to_target, create_matrices=True):
    # Prepare slot duplication definition
    n_flat_slots = np.sum(n_src_to_target)
    slot_i_tgt = np.empty((n_flat_slots,), dtype=np.int64)
    n_dup_next = np.zeros((n_flat_slots,), dtype=np.int64)
    i_slot = 0
    for i_tgt, n in enumerate(n_src_to_target):
        slot_i_tgt[i_slot:i_slot+n] = i_tgt
        if n > 1:
            n_dup_next[i_slot] = n-1
        i_slot += n
    n_dup_next = tuple(n_dup_next)  # For cached version

    # Count combinations for this source node
    count, combinations = count_n_pool_take(len(n_dup_next), n_src, n_dup_next)
    if not create_matrices:
        return count

    # Find nr of connections to slots
    n_taken_combs = np.zeros((count, len(n_src_to_target)), dtype=np.int64)
    for i_tgt, n in enumerate(n_src_to_target):
        if n == 0:
            continue

        conn_comb_tgt = combinations[:, slot_i_tgt == i_tgt]
        n_taken_combs[:, i_tgt] = conn_comb_tgt[:, 0] if n == 1 else np.sum(conn_comb_tgt, axis=1)

    return n_taken_combs


@functools.lru_cache(maxsize=None)
# @numba.njit()
def count_n_pool_take(n_pool, n_take, n_dup_next):
    # Special cases: at the edges of n_take wrt n_pool, we have only 1 combination (duplications don't matter)
    # Example: take 0 from 4 --> 0000; take 4 from 4 --> 1111
    if n_take == 0 or n_take == n_pool:
        combinations = np.zeros((1, n_pool), dtype=np.int64) if n_take == 0 else np.ones((1, n_pool), dtype=np.int64)
        return 1, combinations
    elif n_take < 0 or n_take > n_pool:
        return 0, np.zeros((0, n_pool), dtype=np.int64)

    # We are symmetric wrt n_take, and since the algorithm is a bit faster for lower numbers of n_take, we swap the
    # value if we are higher than half of n_pool
    is_swapped = False
    if n_pool > 1 and n_take > n_pool*.5:
        n_take = n_pool-n_take
        is_swapped = True

    # Get initial combinations matrix
    combinations = _combinations_matrix(n_pool, n_take)

    # Process duplication constraints
    if np.sum(n_dup_next) > 0:
        mask_invalid_dup = np.zeros((combinations.shape[0],), dtype=bool)  # For cached
        # mask_invalid_dup = np.zeros((combinations.shape[0],), dtype=numba.types.bool_)  # For numba
        for i, n_dup in enumerate(n_dup_next):
            if n_dup == 0:
                continue

            # Valid rows have uniformly decreasing or equal number of connections (e.g. 1,1,0, but not 1,0,1 or 0,1,1)
            n_dup_diff = np.diff(combinations[:, i:i+n_dup+1], axis=1)  # For cached
            # n_dup_diff = combinations[:, i+1:i+n_dup+1]-combinations[:, i:i+n_dup]  # For numba
            for j in range(n_dup_diff.shape[1]):
                mask_invalid_dup |= n_dup_diff[:, j] > 0

        # Remove invalid rows
        combinations = combinations[~mask_invalid_dup, :]

    if is_swapped:
        # The consequence of this is that the duplicated slots are filled from the end instead of from the front, but
        # for adding/counting that shouldn't matter
        combinations = (1-combinations)[::-1, :]

    return combinations.shape[0], combinations


@functools.lru_cache(maxsize=None)
# @numba.njit()
def _combinations_matrix(p, r):
    # Based on nump2 of https://stackoverflow.com/a/42202157
    # Prepare combinations matrix and first column
    n0 = p-r+1
    comb_mat = np.zeros((n0, p), dtype=np.int64)
    comb_mat[:n0, :n0] = np.eye(n0)
    n_rep = np.arange(p-r+1)
    # comb_mat[n_rep, n_rep] = 1

    for j in range(1, r):
        # Determine number of repetitions
        reps = (p-r+j) - n_rep
        comb_mat = np.repeat(comb_mat, reps, axis=0)  # For cached
        # comb_mat_rep = np.empty((np.sum(reps), comb_mat.shape[1]), dtype=np.int64)  # For numba
        # i_in_mat = 0
        # for i_rep, n_rep in enumerate(reps):
        #     comb_mat_rep[i_in_mat:i_in_mat+n_rep, :] = comb_mat[i_rep]
        #     i_in_mat += n_rep
        # comb_mat = comb_mat_rep

        ind = np.add.accumulate(reps)  # For cached
        # ind = _add_accumulate(reps)  # For numba

        # Get columns to set to True
        n_rep = np.ones((comb_mat.shape[0],), dtype=np.int64)
        n_rep[ind[:-1]] = 1-reps[1:]
        n_rep[0] = j
        n_rep[:] = np.add.accumulate(n_rep[:])  # For cached
        # n_rep[:] = _add_accumulate(n_rep[:])  # For numba

        comb_mat[np.arange(comb_mat.shape[0]), n_rep] = 1  # For cached
        # for i_rep, i_col in enumerate(n_rep):  # For numba
        #     comb_mat[i_rep, i_col] = True

    return comb_mat


@numba.njit()
def _add_accumulate(arr):
    arr_acc = arr.copy()
    for i in range(1, len(arr)):
        arr_acc[i:] = arr_acc[i-1]+arr[i]
    return arr_acc
