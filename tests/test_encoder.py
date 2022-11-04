import pytest
import numpy as np
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_enc.eager.imputation.first import *


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


class DirectEncoder(EagerEncoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        return np.array([np.arange(0, n_mat)]).T


def test_encoder():
    enc = DirectEncoder(EagerImputer())
    with pytest.raises(RuntimeError):
        enc.get_matrix([0])

    matrix = np.random.randint(0, 3, (10, 2, 3), dtype=int)
    enc = DirectEncoder(EagerImputer(), matrix)
    exist = NodeExistence()
    assert enc.matrix is not matrix
    assert list(enc.matrix.values())[0] is matrix
    assert exist in enc.matrix
    assert enc.matrix[exist] is matrix
    assert enc.n_mat_max == 10

    assert enc._design_vectors[exist].shape == (10, 1)
    assert len(enc.design_vars) == 1
    assert enc.design_vars[0].n_opts == 10

    assert enc.get_n_design_points() == 10
    assert enc.get_imputation_ratio() == 1.

    for _ in range(100):
        val = enc.design_vars[0].get_random()
        assert val >= 0
        assert val < 10

        dv = enc.get_random_design_vector()
        assert len(dv) == 1
        assert dv[0] >= 0
        assert dv[0] < 10

    assert enc.get_matrix_index([0]) == (0, exist)
    assert enc.is_valid_vector([0])
    dv, mat = enc.get_matrix([0])
    assert dv == [0]
    assert np.all(mat == matrix[0, :, :])

    with pytest.raises(NotImplementedError):
        enc.get_matrix([10])

    matrix_mask = np.ones((10,), dtype=bool)
    matrix_mask[0] = False
    with pytest.raises(NotImplementedError):
        enc.get_matrix([0], matrix_mask=matrix_mask)

    matrix_mask = np.zeros((10,), dtype=bool)
    assert not enc.is_valid_vector([0], matrix_mask=matrix_mask)
    dv, mat = enc.get_matrix([0], matrix_mask=matrix_mask)
    assert np.all(dv == [0])
    assert np.all(mat == 0)


def test_encoder_impute():
    matrix = np.random.randint(0, 3, (10, 2, 3), dtype=int)
    enc = DirectEncoder(FirstImputer())
    enc.matrix = matrix
    exist = NodeExistence()
    assert enc.matrix[exist] is matrix
    assert enc.n_mat_max == 10

    assert not enc.is_valid_vector([10])
    assert enc.get_matrix_index([10])[0] is None
    dv, mat = enc.get_matrix([10])
    assert dv == enc._design_vectors[exist][0, :]
    assert np.all(mat == matrix[0, :, :])

    matrix_mask = np.ones((10,), dtype=bool)
    matrix_mask[0] = False
    dv, mat = enc.get_matrix([0], matrix_mask=matrix_mask)
    assert dv == enc._design_vectors[exist][1, :]
    assert np.all(mat == matrix[1, :, :])


class TwoEncoder(EagerEncoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        if n_mat <= 4:
            return np.array([np.arange(0, n_mat)]).T

        n_mat_half = int(np.ceil(n_mat/2))
        dv1 = np.tile(np.arange(0, n_mat_half), 2)[:n_mat]
        dv2 = np.repeat([0, 1], n_mat_half)[:n_mat]
        return np.column_stack([dv1, dv2])


def test_encoder_existence():
    exist1 = NodeExistence()
    exist2 = NodeExistence(src_exists=[True, False])
    matrix_map = {
        exist1: np.random.random((10, 3, 2)),
        exist2: np.random.random((4, 3, 2)),
    }
    enc = TwoEncoder(FirstImputer())
    enc.matrix = matrix_map
    assert enc.n_mat_max == 10

    assert len(enc.design_vars) == 2
    assert enc.design_vars[0].n_opts == 5
    assert enc.design_vars[1].n_opts == 2

    assert enc.is_valid_vector([0, 0], existence=exist1)
    assert enc.is_valid_vector([0, 1], existence=exist1)
    assert enc.is_valid_vector([2, 1], existence=exist1)

    assert enc.get_matrix([0, 0], existence=exist1)

    assert enc.is_valid_vector([0, 0], existence=exist2)
    assert enc.is_valid_vector([2, 0], existence=exist2)
    assert not enc.is_valid_vector([0, 1], existence=exist2)

    assert enc._has_existence(exist1)
    assert not enc._has_existence(NodeExistence(tgt_exists=[True, False]))

    assert enc.get_matrix_index([0, 0], existence=exist1)[0] == 0
    assert enc.get_matrix_index([1, 0], existence=exist2)[0] == 1

    dv, mat = enc.get_matrix([1, 1], existence=exist2)
    assert dv == [1, 0]
    assert np.all(mat == enc._matrix[exist2][1, :, :])


class DuplicateEncoder(EagerEncoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        return np.array([np.repeat(np.arange(0, n_mat), 2)[:n_mat]]).T


def test_duplicate_encoder():
    matrix = np.random.randint(0, 3, (10, 2, 3))
    enc = DuplicateEncoder(EagerImputer())
    with pytest.raises(RuntimeError):
        enc.matrix = matrix


class LowerThanZeroEncoder(EagerEncoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        return np.column_stack([np.arange(0, n_mat)-1, np.arange(0, n_mat)])


class HigherThanZeroEncoder(EagerEncoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        return np.column_stack([np.arange(0, n_mat)+1, np.arange(0, n_mat)+1])


def test_non_zero_encoder():
    matrix = np.random.randint(0, 3, (10, 2, 3))
    with pytest.raises(RuntimeError):
        LowerThanZeroEncoder(FirstImputer(), matrix)
    with pytest.raises(RuntimeError):
        HigherThanZeroEncoder(FirstImputer(), matrix)


def test_flatten_matrix():
    matrix = np.random.randint(0, 3, (10, 4, 3))
    flattened = flatten_matrix(matrix)
    assert flattened.shape == (10, 4*3)


def test_normalize_dvs():
    des_vectors = np.array([
        [0, 1, 0, 2, -1],
        [1, 2, 3, 0,  0],
        [2, 3, 2, 0,  1],
    ])

    assert np.all(EagerEncoder._normalize_design_vectors(des_vectors, remove_gaps=False) == np.array([
        [0, 0, 0, 2, 0],
        [1, 1, 3, 0, 1],
        [2, 2, 2, 0, 2],
    ]))

    assert np.all(EagerEncoder._normalize_design_vectors(des_vectors) == np.array([
        [0, 0, 0, 1, 0],
        [1, 1, 2, 0, 1],
        [2, 2, 1, 0, 2],
    ]))


class DirectZeroEncoder(EagerEncoder):

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        n_mat = matrix.shape[0]
        if n_mat == 1:
            return np.empty((0, 0), dtype=int)
        return np.array([np.arange(0, n_mat)]).T


def test_encoder_zero_dvs():
    src = [Node([1], repeated_allowed=False) for _ in range(2)]
    tgt = [Node([1], repeated_allowed=False) for _ in range(2)]
    exist = NodeExistencePatterns([
        NodeExistence(),
        NodeExistence(src_exists=[True, False], tgt_exists=[True, False]),
    ])
    gen = AggregateAssignmentMatrixGenerator(src, tgt, existence_patterns=exist)
    matrix_map = gen.get_agg_matrix()
    assert len(matrix_map) == 2
    assert matrix_map[exist.patterns[1]].shape[0] == 1

    encoder = DirectZeroEncoder(EagerImputer())
    encoder.matrix = matrix_map

    assert len(encoder.design_vars) == 1
    assert encoder.design_vars[0].n_opts == 2

    dv, mat = encoder.get_matrix([0], existence=exist.patterns[0])
    assert dv == [0]
    assert np.all(mat == np.array([[1, 0], [0, 1]]))
    dv, mat = encoder.get_matrix([0], existence=exist.patterns[1])
    assert dv == [0]
    assert np.all(mat == np.array([[1, 0], [0, 0]]))
