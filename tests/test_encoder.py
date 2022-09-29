import pytest
import numpy as np
from assign_enc.encoding import *


def test_filter_design_vectors():
    design_vectors = np.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 2, 3],
        [0, 1, 2, 4],
    ], dtype=int)

    def _assert_indices(vector, indices):
        assert np.all(np.where(filter_design_vectors(design_vectors, vector))[0] == indices)

    _assert_indices([0, 0, 0, 0], [0])
    _assert_indices([0, 1, 2, 3], [2])
    _assert_indices([0, 1, 1, 0], [])

    _assert_indices([0, 0, None, None], [0])
    _assert_indices([None, 0, None, None], [0])
    _assert_indices([0, 1, None, None], [1, 2, 3])
    _assert_indices([0, 1, 2, None], [2, 3])
    _assert_indices([0, 1, None, 3], [2])


class DirectEncoder(Encoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        return np.array([np.arange(0, n_mat)]).T


def test_encoder():
    enc = DirectEncoder(Imputer())
    with pytest.raises(RuntimeError):
        enc.get_matrix([0])

    matrix = np.random.randint(0, 3, (10, 2, 3), dtype=int)
    enc = DirectEncoder(Imputer(), matrix)
    assert enc.matrix is matrix
    assert enc.n_mat == 10

    assert enc._design_vectors.shape == (10, 1)
    assert len(enc.design_vars) == 1
    assert enc.design_vars[0].n_opts == 10

    for _ in range(100):
        val = enc.design_vars[0].get_random()
        assert val >= 0
        assert val < 10

        dv = enc.get_random_design_vector()
        assert len(dv) == 1
        assert dv[0] >= 0
        assert dv[0] < 10

    dv, mat = enc.get_matrix([0])
    assert dv == [0]
    assert np.all(mat == matrix[0, :, :])

    with pytest.raises(NotImplementedError):
        enc.get_matrix([10])

    matrix_mask = np.ones((10,), dtype=bool)
    matrix_mask[0] = False
    with pytest.raises(NotImplementedError):
        enc.get_matrix([0], matrix_mask=matrix_mask)


def test_encoder_impute():
    matrix = np.random.randint(0, 3, (10, 2, 3), dtype=int)
    enc = DirectEncoder(FirstImputer())
    enc.matrix = matrix
    assert enc.matrix is matrix
    assert enc.n_mat == 10

    dv, mat = enc.get_matrix([10])
    assert dv == enc._design_vectors[0, :]
    assert np.all(mat == matrix[0, :, :])

    matrix_mask = np.ones((10,), dtype=bool)
    matrix_mask[0] = False
    dv, mat = enc.get_matrix([0], matrix_mask=matrix_mask)
    assert dv == enc._design_vectors[1, :]
    assert np.all(mat == matrix[1, :, :])


class DuplicateEncoder(Encoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        return np.array([np.repeat(np.arange(0, n_mat), 2)[:n_mat]]).T


def test_duplicate_encoder():
    matrix = np.random.randint(0, 3, (10, 2, 3), dtype=int)
    enc = DuplicateEncoder(Imputer())
    with pytest.raises(RuntimeError):
        enc.matrix = matrix
