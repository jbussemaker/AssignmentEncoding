import numpy as np
from typing import *

import pytest

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
    matrix = np.random.randint(0, 3, (10, 2, 3), dtype=int)
    enc = DirectEncoder(matrix, Imputer())
    assert enc.matrix is matrix
    assert enc.n_mat == 10

    assert enc._design_vectors.shape == (10, 1)
    assert len(enc.design_vars) == 1
    assert enc.design_vars[0].n_opts == 10

    dv, mat = enc.get_matrix([0])
    assert dv == [0]
    assert np.all(mat == matrix[0, :, :])

    with pytest.raises(NotImplementedError):
        enc.get_matrix([10])

    matrix_mask = np.ones((10,), dtype=bool)
    matrix_mask[0] = False
    with pytest.raises(NotImplementedError):
        enc.get_matrix([0], matrix_mask=matrix_mask)


class DummyImputer(Imputer):

    def impute(self, vector: DesignVector, matrix_mask: MatrixSelectMask) -> Tuple[DesignVector, np.ndarray]:
        design_vectors = self._design_vectors[matrix_mask, :]
        matrices = self._matrix[matrix_mask, :, :]
        return design_vectors[0, :], matrices[0, :, :]


def test_encoder_impute():
    matrix = np.random.randint(0, 3, (10, 2, 3), dtype=int)
    enc = DirectEncoder(matrix, DummyImputer())
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
    with pytest.raises(RuntimeError):
        DuplicateEncoder(matrix, Imputer())
