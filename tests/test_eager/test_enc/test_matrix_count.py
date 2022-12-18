import numpy as np
from assign_enc.matrix import *
from assign_enc.eager.imputation.first import *
from assign_enc.eager.encodings.matrix_count import *


def test_encoding():
    for n in range(15):
        matrix = np.random.randint(0, 3, (n, 3, 4))
        enc = OneVarEncoder(FirstImputer(), matrix)

        assert enc.n_mat_max == n
        if n <= 1:
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


def test_recursive_encoding():
    for n_divide in range(15):
        for n in range(15):
            enc = RecursiveEncoder(FirstImputer(), n_divide=n_divide)
            assert enc.n_divide == max(2, n_divide)
            enc.matrix = np.random.randint(0, 3, (n, 3, 4))

            assert enc.n_mat_max == n
            if n <= 1:
                assert len(enc.design_vars) == 0
                continue
            n_div = max(2, n_divide)
            if n_div >= n:
                assert len(enc.design_vars) == 1
                assert enc.design_vars[0].n_opts == n

                assert enc.get_imputation_ratio() == 1
                assert enc.get_information_index() == (1 if n == 2 else 0)
            else:
                log_ratio = np.log(n)/np.log(n_div)
                assert len(enc.design_vars) == int(np.ceil(log_ratio))
                assert max([dv.n_opts for dv in enc.design_vars]) == n_div

                assert enc.get_imputation_ratio() >= 1
                assert enc.get_information_index() > 0


def test_one_to_one_recursive(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    for n_divide in range(2, 4):
        encoder = RecursiveEncoder(FirstImputer(), n_divide=n_divide)
        encoder.matrix = gen_one_per_existence.get_agg_matrix()
        assert len(encoder.design_vars) == 0

        assert encoder.get_n_design_points() == 1
        assert encoder.get_imputation_ratio() == 1
        assert encoder.get_information_index() == 1
