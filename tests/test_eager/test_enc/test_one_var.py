import numpy as np
from assign_enc.eager.imputation.first import *
from assign_enc.eager.encodings.one_var import *


def test_encoding():
    for n in range(5, 15):
        matrix = np.random.randint(0, 3, (n, 3, 4))
        enc = OneVarEncoder(FirstImputer(), matrix)

        assert enc.n_mat == n
        assert len(enc.design_vars) == 1
        assert enc.design_vars[0].n_opts == n

        assert enc.get_imputation_ratio() == 1.

        for i in range(n):
            dv, mat = enc.get_matrix([i])
            assert dv == [i]
            assert np.all(mat == matrix[i, :, :])