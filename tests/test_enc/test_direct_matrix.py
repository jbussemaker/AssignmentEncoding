import numpy as np
from assign_enc.encoding import *
from assign_enc.encodings.direct_matrix import *


def test_encoding():
    for n in range(5, 15):
        enc = DirectMatrixEncoder(FirstImputer())
        n_src, n_tgt = 3, 4
        enc.matrix = matrix = np.random.randint(0, 3, (n, n_src, n_tgt))

        assert enc.n_mat == n
        assert len(enc.design_vars) == 3*4
        for i in range(n_src):
            for j in range(n_tgt):
                i_dv = i*n_tgt+j
                assert enc.design_vars[i_dv].n_opts == np.max(matrix[:, i, j])-np.min(matrix[:, i, j])+1

        for j in range(n):
            dv = enc.get_random_design_vector()
            imp_dv, mat = enc.get_matrix(dv)
            if np.all(dv == imp_dv):
                assert mat is not None
            else:
                assert np.all(imp_dv == enc._design_vectors[0, :])
                assert np.all(mat == matrix[0, :, :])
