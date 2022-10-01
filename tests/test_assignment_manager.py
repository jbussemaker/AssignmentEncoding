import numpy as np
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_enc.imputation.first import *
from assign_enc.assignment_manager import *


class FlattenEncoder(Encoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat, n_src, n_tgt = matrix.shape
        n_dv = n_src*n_tgt

        # Map matrix elements to design vector values
        design_vectors = matrix.reshape(n_mat, n_dv)

        # Move design vector values so that the first value is always 0
        design_vectors -= np.min(design_vectors, axis=0)
        return design_vectors


def test_manager():
    src = [Node([1, 2]), Node([1])]
    tgt = [Node([0, 1]), Node(min_conn=1)]
    enc = FlattenEncoder(FirstImputer())

    manager = AssignmentManager(src, tgt, enc)
    assert manager.matrix is not None
    assert manager.matrix.shape == (6, 2, 2)

    assert len(manager.design_vars) == 4

    seen_dvs = set()
    for _ in range(100):
        dv = manager.get_random_design_vector()
        imp_dv, conn = manager.get_conns(dv)

        assert imp_dv is not None
        assert conn is not None
        assert len(conn) > 0

        seen_dvs.add(tuple(imp_dv))

        assert np.all(manager.correct_vector(dv) == imp_dv)
        assert np.all(manager.correct_vector(imp_dv) == imp_dv)

    assert len(seen_dvs) < 100
