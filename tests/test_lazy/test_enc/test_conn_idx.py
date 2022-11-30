import numpy as np
from assign_enc.matrix import *
from assign_enc.lazy.encodings.conn_idx import *
from assign_enc.lazy.imputation.constraint_violation import *


def test_encoding():
    encoder = LazyConnIdxMatrixEncoder(LazyConstraintViolationImputer(), FlatConnCombsEncoder())
    src, tgt = [Node([0, 1, 2]) for _ in range(3)], [Node(min_conn=1, repeated_allowed=False) for _ in range(2)]
    encoder.set_nodes(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                      existence_patterns=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [2]})]))
    assert len(encoder.design_vars) == 0

    encoder.set_nodes(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                      existence_patterns=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [1]})]))
    assert len(encoder.design_vars) == 2

    dv, mat = encoder.get_matrix([2, 2], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [2, 2])
    assert np.all(mat == np.array([[1, 0], [0, 1], [0, 1]]))

    dv, mat = encoder.get_matrix([0, 0], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [0, 0])
    assert mat[0, 0] == -1


def test_encoding_transpose():
    encoder = LazyConnIdxMatrixEncoder(LazyConstraintViolationImputer(), FlatConnCombsEncoder(), by_src=False)
    src, tgt = [Node([0, 1, 2]) for _ in range(3)], [Node(min_conn=1, repeated_allowed=False) for _ in range(2)]
    encoder.set_nodes(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                      existence_patterns=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [2]})]))
    assert len(encoder.design_vars) == 2
    _, mat = encoder.get_matrix([0, 0], existence=encoder.existence_patterns.patterns[0])
    assert mat[0, 0] == -1

    encoder.set_nodes(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                      existence_patterns=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [1]})]))
    assert len(encoder.design_vars) > 0

    dv, mat = encoder.get_matrix([0, 2], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [0, 2])
    assert np.all(mat == np.array([[1, 0], [0, 1], [0, 1]]))

    dv, mat = encoder.get_matrix([2, 2], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [2, 2])
    assert mat[0, 0] == -1


def test_encoding_amount_first():
    encoder = LazyConnIdxMatrixEncoder(LazyConstraintViolationImputer(), FlatConnCombsEncoder(), amount_first=True)
    src, tgt = [Node([0, 1, 2]) for _ in range(3)], [Node(min_conn=1, repeated_allowed=False) for _ in range(2)]
    encoder.set_nodes(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                      existence_patterns=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [2]})]))
    assert len(encoder.design_vars) == 0

    encoder.set_nodes(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                      existence_patterns=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [1]})]))
    assert len(encoder.design_vars) == 4

    dv, mat = encoder.get_matrix([0, 0, 0, 0], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [0, 0, 0, 0])
    assert mat[0, 0] == -1

    dv, mat = encoder.get_matrix([1, 1, 1, 1], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [1, 1, 1, 1])
    assert np.all(mat == np.array([[1, 0], [0, 1], [0, 1]]))


def test_encoding_grouped():
    encoder = LazyConnIdxMatrixEncoder(LazyConstraintViolationImputer(), GroupedConnCombsEncoder())
    src, tgt = [Node([0, 1, 2]) for _ in range(3)], [Node(min_conn=1, repeated_allowed=False) for _ in range(2)]
    encoder.set_nodes(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                      existence_patterns=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [2]})]))
    assert len(encoder.design_vars) == 0

    encoder.set_nodes(src=src, tgt=tgt, excluded=[(src[0], tgt[1])],
                      existence_patterns=NodeExistencePatterns([NodeExistence(src_n_conn_override={0: [1]})]))
    assert len(encoder.design_vars) == 4

    dv, mat = encoder.get_matrix([0, 1, 0, 1], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [0, 1, 0, 1])
    assert np.all(mat == np.array([[1, 0], [0, 1], [0, 1]]))

    dv, mat = encoder.get_matrix([0, 0, 0, 0], existence=encoder.existence_patterns.patterns[0])
    assert np.all(dv == [0, 0, 0, 0])
    assert mat[0, 0] == -1
