import numpy as np
from assign_enc.matrix import *
from assign_enc.lazy.imputation.first import *
from assign_enc.lazy.encodings.conn_idx import *
from assign_enc.lazy.encodings.direct_matrix import *


def test_imputer():
    encoder = LazyDirectMatrixEncoder(LazyFirstImputer())
    encoder.set_settings(MatrixGenSettings(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1])]))

    dv, mat = encoder.get_matrix([1, 1])
    assert np.all(dv == [0, 0])
    assert np.all(mat == np.array([[0], [0]]))

    encoder.set_settings(MatrixGenSettings(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1]), Node(min_conn=1)]))

    dv, mat = encoder.get_matrix([1, 1, 1, 1])
    assert np.all(dv == [0, 1, 0, 0])
    assert np.all(mat == np.array([[0, 1], [0, 0]]))


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    g = gen_one_per_existence
    encoder = LazyConnIdxMatrixEncoder(LazyFirstImputer(), FlatConnCombsEncoder())
    encoder.set_settings(g.settings)
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1.2
    assert encoder.get_information_index() == 1

    for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
        dv, mat = encoder.get_matrix([], existence=existence)
        assert dv == []
        assert mat.shape[0] == (len(gen_one_per_existence.src) if i not in [3, 7] else 0)
