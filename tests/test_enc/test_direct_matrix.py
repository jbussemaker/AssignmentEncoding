import numpy as np
from assign_enc.imputation.first import *
from assign_enc.encodings.direct_matrix import *


def test_encoding():
    for n in range(5, 15):
        enc = DirectMatrixEncoder(FirstImputer(), remove_gaps=False)
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

        n_unique = [len(np.unique(enc._design_vectors[:, i_dv])) for i_dv in range(enc._design_vectors.shape[1])]

        enc_remove_gaps = DirectMatrixEncoder(FirstImputer())
        enc_remove_gaps.matrix = matrix
        n_unique_rg = [len(np.unique(enc_remove_gaps._design_vectors[:, i_dv]))
                       for i_dv in range(enc_remove_gaps._design_vectors.shape[1])]
        assert np.all(n_unique_rg == n_unique)
        assert all([dv.n_opts == n_unique_rg[i] for i, dv in enumerate(enc_remove_gaps.design_vars)])
