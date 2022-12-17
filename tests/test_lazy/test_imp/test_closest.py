import numpy as np
from assign_enc.matrix import *
from assign_enc.lazy.imputation.closest import *
from assign_enc.lazy.encodings.conn_idx import *
from assign_enc.lazy.encodings.direct_matrix import *


def test_imputer():
    encoder = LazyDirectMatrixEncoder(LazyClosestImputer())
    encoder.set_nodes(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)])
    assert encoder._matrix_gen.max_conn == 3

    dv = encoder.design_vars
    assert [d.n_opts for d in dv] == [2, 3, 2, 4]

    dv, mat = encoder.get_matrix([1, 2, 1, 0])
    assert np.all(dv == [0, 2, 1, 0])
    assert np.all(mat == np.array([[0, 2], [1, 0]]))


def test_max_tries():
    encoder = LazyDirectMatrixEncoder(LazyClosestImputer(n_max_tries=100))
    encoder.set_nodes(src=[Node([1], repeated_allowed=False)],
                      tgt=[Node([0, 1], repeated_allowed=False) for _ in range(10)])
    assert encoder.get_imputation_ratio() > 100

    n_invalid = 0
    for _ in range(5):
        dv_rand = encoder.get_random_design_vector()
        dv, mat = encoder.get_matrix(dv_rand)
        if mat[0, 0] == -1:
            assert np.all(dv == dv_rand)
            assert np.all(mat == -1)
            n_invalid += 1
    assert n_invalid > 0


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    g = gen_one_per_existence
    encoder = LazyConnIdxMatrixEncoder(LazyClosestImputer(), FlatConnCombsEncoder())
    encoder.set_nodes(g.src, g.tgt, existence_patterns=g.existence_patterns)
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1
    assert encoder.get_information_index() == 1

    for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
        dv, mat = encoder.get_matrix([], existence=existence)
        assert dv == []
        assert mat.shape[0] == (len(gen_one_per_existence.src) if i not in [3, 7] else 0)
