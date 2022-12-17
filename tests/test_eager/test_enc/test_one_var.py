import numpy as np
from assign_enc.matrix import *
from assign_enc.eager.imputation.first import *
from assign_enc.eager.encodings.one_var import *


def test_encoding():
    for n in range(1, 15):
        matrix = np.random.randint(0, 3, (n, 3, 4))
        enc = OneVarEncoder(FirstImputer(), matrix)

        assert enc.n_mat_max == n
        if n == 1:
            assert len(enc.design_vars) == 0
            continue
        assert len(enc.design_vars) == 1
        assert enc.design_vars[0].n_opts == n

        assert enc.get_imputation_ratio() == 1.

        for i in range(n):
            dv, mat = enc.get_matrix([i])
            assert dv == [i]
            assert np.all(mat == matrix[i, :, :])


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    encoder = OneVarEncoder(FirstImputer())
    encoder.matrix = gen_one_per_existence.get_agg_matrix()
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1
    assert encoder.get_information_index() == 1
