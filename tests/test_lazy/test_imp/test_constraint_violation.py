import numpy as np
from assign_enc.matrix import *
from assign_enc.assignment_manager import *
from assign_enc.lazy.encodings.direct_matrix import *
from assign_enc.lazy.imputation.constraint_violation import *


def test_imputer():
    encoder = LazyDirectMatrixEncoder(LazyConstraintViolationImputer())
    encoder.set_nodes(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1])])

    dv, mat = encoder.get_matrix([0, 1])
    assert np.all(dv == [0, 1])
    assert np.all(mat == np.array([[0], [1]]))

    dv, mat = encoder.get_matrix([1, 1])
    assert np.all(dv == [1, 1])
    assert np.all(mat == np.array([[-1], [-1]]))


def test_assignment_manager():
    encoder = LazyDirectMatrixEncoder(LazyConstraintViolationImputer())
    manager = LazyAssignmentManager(src=[Node([1], repeated_allowed=False) for _ in range(2)],
                                    tgt=[Node([1], repeated_allowed=False) for _ in range(2)], encoder=encoder)
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
