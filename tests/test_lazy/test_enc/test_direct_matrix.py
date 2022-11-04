import numpy as np
from assign_enc.matrix import *
from assign_enc.lazy.encodings.direct_matrix import *
from assign_enc.lazy.imputation.constraint_violation import *


def test_encoding():
    encoder = LazyDirectMatrixEncoder(LazyConstraintViolationImputer())
    encoder.set_nodes(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)])
    assert encoder._matrix_gen.max_conn == 3
    assert encoder._matrix_gen.count_all_matrices() == 21

    dv = encoder.design_vars
    assert len(dv) == 4
    assert [d.n_opts for d in dv] == [2, 3, 2, 4]

    _, mat = encoder.get_matrix([0, 1, 0, 0])
    assert np.all(mat == np.array([[0, 1], [0, 0]]))

    _, mat = encoder.get_matrix([0, 1, 0, 1])
    assert np.all(mat == np.array([[0, 1], [0, 1]]))

    _, mat = encoder.get_matrix([0, 2, 1, 1])
    assert np.all(mat == np.array([[0, 2], [1, 1]]))

    assert encoder.get_n_design_points() == 48
    assert encoder.get_imputation_ratio() == 48/21


def test_encoder_excluded():
    encoder = LazyDirectMatrixEncoder(LazyConstraintViolationImputer())
    src = [Node([0, 1, 2]), Node(min_conn=0)]
    tgt = [Node([0, 1]), Node(min_conn=1)]
    encoder.set_nodes(src=src, tgt=tgt, excluded=[(src[1], tgt[0])])
    assert encoder._matrix_gen.max_conn == 3

    dv = encoder.design_vars
    assert len(dv) == 3
    assert [d.n_opts for d in dv] == [2, 3, 4]

    _, mat = encoder.get_matrix([0, 1, 0])
    assert np.all(mat == np.array([[0, 1], [0, 0]]))

    _, mat = encoder.get_matrix([0, 2, 1])
    assert np.all(mat == np.array([[0, 2], [0, 1]]))


def test_encoder_no_repeat():
    encoder = LazyDirectMatrixEncoder(LazyConstraintViolationImputer())
    src = [Node([0, 1, 2]), Node(min_conn=0, repeated_allowed=False)]
    tgt = [Node([0, 1], repeated_allowed=False), Node(min_conn=1)]
    encoder.set_nodes(src=src, tgt=tgt)
    assert encoder._matrix_gen.max_conn == 3

    dv = encoder.design_vars
    assert len(dv) == 4
    assert [d.n_opts for d in dv] == [2, 3, 2, 2]

    _, mat = encoder.get_matrix([0, 1, 0, 0])
    assert np.all(mat == np.array([[0, 1], [0, 0]]))

    _, mat = encoder.get_matrix([0, 2, 0, 1])
    assert np.all(mat == np.array([[0, 2], [0, 1]]))


def test_encoder_existence():
    encoder = LazyDirectMatrixEncoder(LazyConstraintViolationImputer())
    src = [Node([0, 1, 2]), Node(min_conn=0, repeated_allowed=False)]
    tgt = [Node([0, 1], repeated_allowed=False), Node(min_conn=1)]
    exist = NodeExistencePatterns([
        NodeExistence(),
        NodeExistence(tgt_exists=[True, False]),
    ])
    encoder.set_nodes(src, tgt, existence_patterns=exist)

    assert len(encoder._existence_design_vars) == 2
    assert len(encoder._existence_design_vars[exist.patterns[0]]) == 4
    assert len(encoder._existence_design_vars[exist.patterns[1]]) == 2

    dv = encoder.design_vars
    assert len(dv) == 4
    assert [d.n_opts for d in dv] == [2, 3, 2, 2]

    _, mat = encoder.get_matrix([0, 1, 0, 0])
    assert np.all(mat == np.array([[0, 1], [0, 0]]))
    _, mat = encoder.get_matrix([0, 1, 0, 0], existence=NodeExistence())
    assert np.all(mat == np.array([[0, 1], [0, 0]]))
    _, mat = encoder.get_matrix([0, 1, 0, 0], existence=exist.patterns[0])
    assert np.all(mat == np.array([[0, 1], [0, 0]]))

    _, mat = encoder.get_matrix([1, 0, 0, 0], existence=exist.patterns[0])
    assert np.all(mat == np.array([[-1, -1], [-1, -1]]))
    _, mat = encoder.get_matrix([1, 0, 0, 0], existence=exist.patterns[1])
    assert np.all(mat == np.array([[1, 0], [0, 0]]))
