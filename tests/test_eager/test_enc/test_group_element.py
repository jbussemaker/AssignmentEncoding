import numpy as np
from assign_enc.eager.imputation.first import *
from assign_enc.eager.encodings.group_element import *


def test_element_grouper():
    matrix = np.array([
        [[0, 1, 0],
         [1, 1, 0]],
        [[2, 0, 0],
         [1, 1, 2]],
        [[1, 1, 0],
         [1, 1, 2]],
        [[1, 1, 0],
         [1, 2, 2]],
    ])

    encoder = ElementGroupedEncoder(FirstImputer(), matrix)
    encoder.matrix = matrix

    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, 0],
        [2, 0],
        [1, 0],
        [1, 1],
    ]))

    assert [dv.n_opts for dv in encoder.design_vars] == [3, 2]
    assert encoder.get_n_design_points() == 3*2
    assert encoder.get_imputation_ratio() == 1.5

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(list(encoder._design_vectors.values())[0] == np.array([
        [0, 1, 0],
        [2, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]))
