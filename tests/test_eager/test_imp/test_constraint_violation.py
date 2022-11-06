import numpy as np
from assign_enc.matrix import *
from assign_enc.assignment_manager import *
from assign_enc.eager.encodings.direct_matrix import *
from assign_enc.eager.imputation.constraint_violation import *


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


def test_assignment_manager():
    encoder = DirectMatrixEncoder(ConstraintViolationImputer())
    manager = AssignmentManager(src=[Node([1], repeated_allowed=False) for _ in range(2)],
                                tgt=[Node([1], repeated_allowed=False) for _ in range(2)], encoder=encoder)
    assert manager.encoder.n_mat_max == 2
    assert len(manager.design_vars) == 4

    dv, mat = manager.get_matrix([1, 0, 1, 0])
    assert np.all(dv == [1, 0, 1, 0])
    assert np.all(mat == -1)

    dv, conn_idx = manager.get_conn_idx([1, 0, 1, 0])
    assert np.all(dv == [1, 0, 1, 0])
    assert conn_idx is None
    dv, conns = manager.get_conns([1, 0, 1, 0])
    assert np.all(dv == [1, 0, 1, 0])
    assert conns is None
