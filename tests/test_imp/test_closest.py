import numpy as np
from assign_enc.eager.imputation.closest import *
from assign_enc.eager.encodings.direct_matrix import *


def test_imputer():
    matrix = np.array([
        [[0, 0, 0, 0]],
        [[0, 1, 0, 0]],
        [[1, 0, 0, 0]],
        [[1, 1, 0, 0]],
        [[1, 1, 1, 1]],
        [[1, 1, 1, 2]],
    ])
    encoder = DirectMatrixEncoder(ClosestImputer(), matrix)
    dvs = encoder._design_vectors
    assert dvs.shape == (6, 4)

    assert np.all(ClosestImputer._calc_dist_manhattan(dvs, np.array([0, 0, 0, 0])) == np.array([0, 1, 1, 2, 4, 5]))
    assert np.all(ClosestImputer._calc_dist_manhattan(dvs, np.array([1, 1, 2, 1])) == np.array([5, 4, 4, 3, 1, 2]))

    assert np.all(ClosestImputer._calc_dist_euclidean(dvs, np.array([0, 0, 0, 0])) ==
                  np.array([0, 1, 1, np.sqrt(2), np.sqrt(4), np.sqrt(7)]))


def test_imputer_euclidean():
    matrix = np.array([
        [[0, 0, 0, 0]],
        [[0, 1, 0, 0]],
        [[1, 0, 0, 0]],
        [[1, 1, 0, 0]],
        [[1, 1, 1, 1]],
        [[1, 1, 1, 2]],
    ])
    encoder = DirectMatrixEncoder(ClosestImputer(), matrix)

    dv, mat = encoder.get_matrix([0, 0, 0, 0])
    assert np.all(dv == [0, 0, 0, 0])
    assert np.all(mat == matrix[0, :, :])

    dv, mat = encoder.get_matrix([0, 0, 1, 0])
    assert np.all(dv == [0, 0, 0, 0])
    assert np.all(mat == matrix[0, :, :])

    dv, mat = encoder.get_matrix([1, 1, 1, 0])
    assert np.all(dv == [1, 1, 0, 0])
    assert np.all(mat == matrix[3, :, :])

    dv, mat = encoder.get_matrix([1, 1, 2, 0])
    assert np.all(dv == [1, 1, 1, 1])
    assert np.all(mat == matrix[4, :, :])


def test_imputer_manhattan():
    matrix = np.array([
        [[0, 0, 0, 0]],
        [[0, 1, 0, 0]],
        [[1, 0, 0, 0]],
        [[1, 1, 0, 0]],
        [[1, 1, 1, 1]],
        [[1, 1, 1, 2]],
    ])
    encoder = DirectMatrixEncoder(ClosestImputer(euclidean=False), matrix)

    dv, mat = encoder.get_matrix([0, 0, 0, 0])
    assert np.all(dv == [0, 0, 0, 0])
    assert np.all(mat == matrix[0, :, :])

    dv, mat = encoder.get_matrix([0, 0, 1, 0])
    assert np.all(dv == [0, 0, 0, 0])
    assert np.all(mat == matrix[0, :, :])

    dv, mat = encoder.get_matrix([1, 1, 1, 0])
    assert np.all(dv == [1, 1, 0, 0])
    assert np.all(mat == matrix[3, :, :])

    dv, mat = encoder.get_matrix([1, 1, 2, 0])
    assert np.all(dv == [1, 1, 0, 0])
    assert np.all(mat == matrix[3, :, :])


def test_imputer_masked():
    matrix = np.array([
        [[0, 0, 0, 0]],
        [[0, 1, 0, 0]],
        [[1, 0, 0, 0]],
        [[1, 1, 0, 0]],
        [[1, 1, 1, 1]],
        [[1, 1, 1, 2]],
    ])
    encoder = DirectMatrixEncoder(ClosestImputer(), matrix)
    mask = np.array([False, True, True, True, False, True], dtype=bool)

    dv, mat = encoder.get_matrix([0, 0, 0, 0], matrix_mask=mask)
    assert np.all(dv == [0, 1, 0, 0])
    assert np.all(mat == matrix[1, :, :])

    dv, mat = encoder.get_matrix([1, 1, 2, 0], matrix_mask=mask)
    assert np.all(dv == [1, 1, 0, 0])
    assert np.all(mat == matrix[3, :, :])


def test_imputer_random():
    for euclidean in [True, False]:
        matrix = np.random.randint(0, 3, (10, 3, 4))
        encoder = DirectMatrixEncoder(ClosestImputer(euclidean=euclidean), matrix)

        n_imp = 0
        for _ in range(100):
            dv = encoder.get_random_design_vector()
            if not encoder.is_valid_vector(dv):
                n_imp += 1
                dv_imp, mat = encoder.get_matrix(dv)
                assert np.any(dv != dv_imp)
                assert mat.shape == (3, 4)

        assert n_imp > 0
