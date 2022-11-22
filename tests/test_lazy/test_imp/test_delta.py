import numpy as np
from assign_enc.matrix import *
from assign_enc.lazy.imputation.delta import *
from assign_enc.lazy.encodings.direct_matrix import *


def test_imputer():
    encoder = LazyDirectMatrixEncoder(LazyDeltaImputer())
    encoder.set_nodes(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)])
    assert encoder._matrix_gen.max_conn == 3

    dv = encoder.design_vars
    assert [d.n_opts for d in dv] == [2, 3, 2, 4]

    dv, mat = encoder.get_matrix([1, 2, 1, 0])
    assert np.all(dv == [1, 1, 0, 0])
    assert np.all(mat == np.array([[1, 1], [0, 0]]))


def test_max_tries():
    encoder = LazyDirectMatrixEncoder(LazyDeltaImputer(n_max_tries=10000))
    encoder.set_nodes(src=[Node([1], repeated_allowed=False)],
                      tgt=[Node([0, 1], repeated_allowed=False) for _ in range(20)])
    assert encoder.get_imputation_ratio() > 50000

    n_invalid = 0
    for _ in range(5):
        dv_rand = encoder.get_random_design_vector()
        dv, mat = encoder.get_matrix(dv_rand)
        if mat[0, 0] == -1:
            assert np.all(dv == dv_rand)
            assert np.all(mat == -1)
            n_invalid += 1
    assert n_invalid > 0
