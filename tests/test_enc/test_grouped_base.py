import pytest
import numpy as np
from assign_enc.imputation.first import *
from assign_enc.encodings.grouped_base import *


class HalfGroupedEncoder(GroupedEncoder):

    def _get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray) -> np.ndarray:
        grouping_values = np.zeros((sub_matrix.shape[0],), dtype=int)
        n_half = len(grouping_values) // 2
        grouping_values[n_half:] = 1
        return grouping_values


def test_half_grouped_encoder():
    matrix = np.random.randint(0, 3, (11, 4, 3))
    encoder = HalfGroupedEncoder(FirstImputer(), matrix)
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    ]).T)
    assert len(encoder.design_vars) == 4
    assert all([dv.n_opts == 2 for dv in encoder.design_vars])


class EvenNonUniqueGrouper(GroupedEncoder):

    def _get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray) -> np.ndarray:
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
