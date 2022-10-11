import numpy as np
from assign_enc.matrix import *
from assign_enc.lazy.imputation.first import *
from assign_enc.lazy.encodings.direct_matrix import *


def test_imputer():
    encoder = LazyDirectMatrixEncoder(LazyFirstImputer())
    encoder.set_nodes(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1])])

    dv, mat = encoder.get_matrix([1, 1])
    assert np.all(dv == [0, 0])
    assert np.all(mat == np.array([[0], [0]]))

    encoder.set_nodes(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1]), Node(min_conn=1)])

    dv, mat = encoder.get_matrix([1, 1, 1, 1])
    assert np.all(dv == [0, 0, 0, 1])
    assert np.all(mat == np.array([[0, 0], [0, 1]]))
