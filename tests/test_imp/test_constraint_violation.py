import numpy as np
from assign_enc.encodings.direct_matrix import *
from assign_enc.imputation.constraint_violation import *


def test_imputer():
    matrix = np.random.randint(0, 3, (10, 3, 4))
    encoder = DirectMatrixEncoder(ConstraintViolationImputer(), matrix)

    n_imp = 0
    for _ in range(100):
        dv = encoder.get_random_design_vector()
        if not encoder.is_valid_vector(dv):
            n_imp += 1
            dv_imp, mat = encoder.get_matrix(dv)
            assert np.all(dv == dv_imp)
            assert mat.shape == (3, 4)
            assert np.all(mat == -1)

    assert n_imp > 0
