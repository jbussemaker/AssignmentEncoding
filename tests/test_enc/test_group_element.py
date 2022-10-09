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

    assert np.all(encoder._design_vectors == np.array([
        [0, 0],
        [2, 0],
        [1, 0],
        [1, 1],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(encoder._design_vectors == np.array([
        [0, 1, 0],
        [2, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ]))
