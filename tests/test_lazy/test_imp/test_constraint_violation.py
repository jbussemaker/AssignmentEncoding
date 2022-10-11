import numpy as np
from assign_enc.matrix import *
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
