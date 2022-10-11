import itertools
import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.lazy_encoding import *
from assign_enc.assignment_manager import *


class DummyImputer(LazyImputer):

    def _impute(self, vector: DesignVector, matrix: np.ndarray, src_exists: np.ndarray, tgt_exists: np.ndarray,
                validate: Callable[[np.ndarray], bool], tried_vectors: Set[Tuple[int, ...]]) -> Tuple[DesignVector, np.ndarray]:
        assert not validate(matrix)
        assert tuple(vector) in tried_vectors
        imputed_dv = np.array(vector)*0
        imputed_matrix = matrix*0-1
        return imputed_dv, imputed_matrix


class DummyLazyEncoder(LazyEncoder):

    def __init__(self, imputer, n_opts=2):
        self._n_opts = n_opts
        super().__init__(imputer)

    def _encode(self) -> List[DiscreteDV]:
        n_dv = len(self.src)*len(self.tgt)
        return [DiscreteDV(n_opts=self._n_opts) for _ in range(n_dv)]

    def _decode(self, vector: DesignVector, src_exists: np.ndarray, tgt_exists: np.ndarray) -> np.ndarray:
        return np.reshape(np.array(vector), (self.n_src, self.n_tgt))


def test_lazy_encoder():
    src, tgt = [Node([0, 1]), Node([0, 1])], [Node([0, 1]), Node([0, 1])]
    enc = DummyLazyEncoder(DummyImputer())
    enc.set_nodes(src, tgt)
    assert enc.src == src
    assert enc.tgt == tgt
    assert enc.n_src == 2
    assert enc.n_tgt == 2
    assert enc.ex is None

    dv = enc.design_vars
    assert len(dv) == 4
    assert all([d.n_opts == 2 for d in dv])

    assert np.all(enc._decode([0, 1, 1, 1], np.array([True]*2), np.array([True]*2)) == np.array([[0, 1], [1, 1]]))

    matrix, _, _ = enc._decode_vector([0, 1, 1, 1])
    assert np.all(matrix == np.array([[0, 1], [1, 1]]))

    assert not enc.is_valid_vector([0, 1, 1, 1])
    assert enc.is_valid_vector([0, 1, 1, 0])
    assert enc.is_valid_vector([1, 0, 0, 1])

    dv, matrix = enc.get_matrix([0, 1, 1, 0])
    assert np.all(dv == [0, 1, 1, 0])
    assert np.all(matrix == np.array([[0, 1], [1, 0]]))

    assert enc.get_conn_idx(matrix) == [(0, 1), (1, 0)]
    assert enc.get_conns(matrix) == [(src[0], tgt[1]), (src[1], tgt[0])]

    dv, matrix = enc.get_matrix([0, 1, 1, 1])
    assert np.all(dv == [0, 0, 0, 0])
    assert np.all(matrix == -1)


def test_lazy_imputer_none_exist():
    enc = DummyLazyEncoder(DummyImputer())
    enc.set_nodes(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1]), Node([0, 1])])

    matrix, _, _ = enc._decode_vector([0, 1, 1, 0])
    assert np.all(matrix == np.array([[0, 1], [1, 0]]))

    dv, matrix = enc.get_matrix([0, 1, 1, 0], src_exists=[False, False], tgt_exists=[False, False])
    assert np.all(dv == [0, 0, 0, 0])
    assert np.all(matrix == 0)

    dv, matrix = enc.get_matrix(np.array([0, 1, 1, 0]), src_exists=[False, False], tgt_exists=[False, False])
    assert np.all(dv == [0, 0, 0, 0])
    assert np.all(matrix == 0)


def test_lazy_imputer_no_exist_correct():
    enc = DummyLazyEncoder(DummyImputer())
    enc.set_nodes(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1]), Node([0, 1])])

    dv, matrix = enc.get_matrix([0, 1, 1, 1], src_exists=[True, False])
    assert np.all(dv == [0, 1, 1, 1])
    assert np.all(matrix == np.array([[0, 1], [0, 0]]))

    dv, matrix = enc.get_matrix([0, 1, 1, 1], tgt_exists=[True, False])
    assert np.all(dv == [0, 1, 1, 1])
    assert np.all(matrix == np.array([[0, 0], [1, 0]]))


def test_lazy_imputer_cache():
    enc = DummyLazyEncoder(DummyImputer())
    enc.set_nodes(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1]), Node([0, 1])])

    def _stop_impute(*args, **kwargs):
        raise RuntimeError

    for _ in range(2):
        n_imputed = 0
        for dv in itertools.product(*[list(range(dv.n_opts)) for dv in enc.design_vars]):
            dv = list(dv)
            imp_dv, matrix = enc.get_matrix(dv)
            if matrix[0, 0] == -1:
                n_imputed += 1

        assert len(enc._imputer._impute_cache) == n_imputed
        enc._imputer._impute = _stop_impute


def test_lazy_assignment_manager():
    src = [Node([0, 1]), Node([0, 1])]
    tgt = [Node([0, 1]), Node([0, 1])]
    manager = LazyAssignmentManager(src, tgt, DummyLazyEncoder(DummyImputer()))

    dv = manager.design_vars
    assert len(dv) == 4
    assert all([d.n_opts == 2 for d in dv])

    dv, matrix = manager.get_matrix([0, 1, 1, 0])
    assert np.all(dv == [0, 1, 1, 0])
    assert np.all(matrix == np.array([[0, 1], [1, 0]]))

    assert manager.correct_vector([0, 1, 1, 0]) == dv
    assert np.all(manager.correct_vector([0, 1, 1, 1]) == [0, 0, 0, 0])

    _, conn_idx = manager.get_conn_idx([0, 1, 1, 0])
    assert conn_idx == [(0, 1), (1, 0)]
    _, conn_idx = manager.get_conn_idx([0, 1, 1, 1])
    assert not conn_idx
