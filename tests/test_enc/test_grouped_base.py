import pytest
import itertools
import numpy as np
from assign_enc.eager.imputation.first import *
from assign_enc.eager.encodings.grouped_base import *


class ProductEncoder(GroupedEncoder):

    def __init__(self, *args, **kwargs):
        self._dvs = None
        super().__init__(*args, **kwargs)

    def _prepare_grouping(self, matrix: np.ndarray):
        n_var = int(np.ceil(np.log2(matrix.shape[0])))
        design_vectors = np.array(list(itertools.product(*[[0, 1] for _ in range(n_var)]))[:matrix.shape[0]])
        for i_var in reversed(range(1, n_var)):
            design_vectors[:, i_var:] += np.array([2*design_vectors[:, i_var-1]]).T
        self._dvs = design_vectors

    def _get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        if dv_idx >= self._dvs.shape[1]:
            return np.zeros((sub_matrix.shape[0],))
        return self._dvs[i_sub_matrix, dv_idx]


def test_half_grouped_encoder():
    matrix = np.random.randint(0, 3, (11, 4, 3))
    encoder = ProductEncoder(FirstImputer(), matrix)
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    ]).T)
    assert len(encoder.design_vars) == 4
    assert all([dv.n_opts == 2 for dv in encoder.design_vars])


def test_no_group_normalize():
    matrix = np.random.randint(0, 3, (11, 4, 3))
    encoder = ProductEncoder(FirstImputer(), matrix, normalize_within_group=False)
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2],
        [0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3],
        [0, 1, 2, 3, 2, 3, 4, 5, 2, 3, 4],
    ]).T)
    assert len(encoder.design_vars) == 4
    assert [dv.n_opts for dv in encoder.design_vars] == [2, 3, 4, 6]


class EvenNonUniqueGrouper(GroupedEncoder):

    def _get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        grouping_values = np.zeros((sub_matrix.shape[0],), dtype=int)
        if dv_idx % 2 == 0:
            return grouping_values
        n_half = len(grouping_values) // 2
        grouping_values[n_half:] = 1
        return grouping_values


def test_skip_des_var():
    matrix = np.random.randint(0, 3, (11, 4, 3))
    encoder = EvenNonUniqueGrouper(FirstImputer(), matrix=matrix)
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    ]).T)


def test_index_grouped_encoder():
    matrix = np.random.randint(0, 3, (11, 4, 3))
    encoder = GroupByIndexEncoder(FirstImputer(), n_groups=2, matrix=matrix)
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    ]).T)

    encoder = GroupByIndexEncoder(FirstImputer(), n_groups=3, matrix=matrix)
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
    ]).T)

    encoder = GroupByIndexEncoder(FirstImputer(), n_groups=4, matrix=matrix)
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3],
        [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 3],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]).T)

    encoder = GroupByIndexEncoder(FirstImputer())
    encoder.n_groups = 1
    with pytest.raises(RuntimeError):
        encoder.matrix = matrix


class PreprocessedIndexGroupedEncoder(GroupedEncoder):

    def __init__(self, *args, **kwargs):
        self._preprocessed_dvs = None
        super().__init__(*args, **kwargs)

    def _prepare_grouping(self, matrix: np.ndarray):
        n_mat = matrix.shape[0]
        self._preprocessed_dvs = des_var_values = np.zeros((n_mat, 2))

        n_half = n_mat // 2
        des_var_values[n_half:, 0] = 1
        des_var_values[:n_half, 1] = np.arange(0, n_half)
        des_var_values[n_half:, 1] = np.arange(0, n_mat-n_half)

    def _get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        return self._preprocessed_dvs[i_sub_matrix, dv_idx]


def test_grouped_encoder_by_sub_matrix_idx():
    matrix = np.random.randint(0, 3, (11, 4, 3))
    encoder = PreprocessedIndexGroupedEncoder(FirstImputer(), matrix)
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5],
    ]).T)
