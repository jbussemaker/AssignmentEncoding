import pytest
import numpy as np
from assign_enc.matrix import *
from assign_enc.eager.imputation.first import *
from assign_enc.eager.encodings.direct_matrix import *


def test_encoding():
    n_checked = 0
    for n in range(5, 15):
        enc = DirectMatrixEncoder(FirstImputer(), remove_gaps=False)
        n_src, n_tgt = 3, 4
        enc.matrix = matrix = np.random.randint(0, 3, (n, n_src, n_tgt))

        assert enc.n_mat_max == n
        if len(enc.design_vars) != 3*4:
            continue
        n_checked += 1
        assert len(enc.design_vars) == 3*4
        n_des_points = np.cumprod([dv.n_opts for dv in enc.design_vars])[-1]
        assert enc.get_imputation_ratio() == n_des_points/n
        assert enc.get_imputation_ratio() > 7000

        for i in range(n_src):
            for j in range(n_tgt):
                i_dv = i*n_tgt+j
                assert enc.design_vars[i_dv].n_opts == np.max(matrix[:, i, j])-np.min(matrix[:, i, j])+1

        enc_dvs = list(enc._design_vectors.values())[0]
        for j in range(n):
            dv = enc.get_random_design_vector()
            imp_dv, mat = enc.get_matrix(dv)
            if np.all(dv == imp_dv):
                assert mat is not None
            else:
                assert np.all(imp_dv == enc_dvs[0, :])
                assert np.all(mat == matrix[0, :, :])

        n_unique = [len(np.unique(enc_dvs[:, i_dv])) for i_dv in range(enc_dvs.shape[1])]

        enc_remove_gaps = DirectMatrixEncoder(FirstImputer())
        enc_remove_gaps.matrix = matrix
        enc_dvs = list(enc_remove_gaps._design_vectors.values())[0]
        n_unique_rg = [len(np.unique(enc_dvs[:, i_dv]))
                       for i_dv in range(enc_dvs.shape[1])]
        assert np.all(n_unique_rg == n_unique)
        assert all([dv.n_opts == n_unique_rg[i] for i, dv in enumerate(enc_remove_gaps.design_vars)])
    assert n_checked > 0


def test_encoder_excluded():
    src = [Node([0, 1, 2]), Node(min_conn=0)]
    tgt = [Node([0, 1]), Node(min_conn=1)]
    matrix_gen = AggregateAssignmentMatrixGenerator(src=src, tgt=tgt, excluded=[(src[1], tgt[0])])
    encoder = DirectMatrixEncoder(FirstImputer())
    encoder.matrix = matrix = matrix_gen.get_agg_matrix()
    assert list(matrix.values())[0].shape[0] == 14

    assert len(encoder.design_vars) == 3
    assert encoder.get_n_design_points() == 2*3*4
    assert encoder.get_imputation_ratio() == 24/14

    dv = encoder.design_vars
    assert len(dv) == 3
    assert [d.n_opts for d in dv] == [2, 3, 4]

    _, mat = encoder.get_matrix([0, 1, 0])
    assert np.all(mat == np.array([[0, 1], [0, 0]]))

    _, mat = encoder.get_matrix([0, 2, 1])
    assert np.all(mat == np.array([[0, 2], [0, 1]]))


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    encoder = DirectMatrixEncoder(FirstImputer())
    encoder.matrix = gen_one_per_existence.get_agg_matrix()
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1
    assert encoder.get_information_index() == 1

    for i, existence in enumerate(gen_one_per_existence.existence_patterns.patterns):
        dv, mat = encoder.get_matrix([], existence=existence)
        assert dv == []
        assert mat.shape[0] == (len(gen_one_per_existence.src) if i not in [3, 7] else 0)
