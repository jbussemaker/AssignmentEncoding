import numpy as np
from assign_enc.encoding import *
from assign_enc.eager.imputation.first import *
from assign_enc.eager.encodings import *


def test_encoder():
    for _ in range(10):
        matrix = np.random.randint(0, 5, (10, 4, 3))
        n_tot = np.sum(flatten_matrix(matrix), axis=1)

        encoder = AmountFirstGroupedEncoder(FirstImputer(), TotalAmountGrouper(), OneVarLocationGrouper())
        encoder.matrix = matrix

        assert len(encoder.design_vars) == 2
        assert encoder.get_imputation_ratio() > 1.2

        n_tot_unique = np.unique(n_tot)
        assert encoder.design_vars[0].n_opts == len(n_tot_unique)
        n_tot_max = max([np.sum(n_tot == n_val) for n_val in n_tot_unique])
        assert encoder.design_vars[1].n_opts == n_tot_max


def test_source_amount_grouper():
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

    encoder = AmountFirstGroupedEncoder(FirstImputer(), SourceAmountGrouper(), OneVarLocationGrouper())
    encoder.matrix = matrix

    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 2, 0],
    ]))


def test_source_amount_flat_grouper():
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

    encoder = AmountFirstGroupedEncoder(FirstImputer(), SourceAmountFlattenedGrouper(), OneVarLocationGrouper())
    encoder.matrix = matrix

    assert np.all(encoder._design_vectors == np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [2, 0],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(encoder._design_vectors == np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [2, 0],
    ]))


def test_target_amount_grouper():
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

    encoder = AmountFirstGroupedEncoder(FirstImputer(), TargetAmountGrouper(), OneVarLocationGrouper())
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
        [2, 0, 1],
        [1, 1, 1],
        [1, 2, 1],
    ]))


def test_target_amount_flat_grouper():
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

    encoder = AmountFirstGroupedEncoder(FirstImputer(), TargetAmountFlattenedGrouper(), OneVarLocationGrouper())
    encoder.matrix = matrix

    assert np.all(encoder._design_vectors == np.array([
        [0],
        [3],
        [1],
        [2],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(encoder._design_vectors == np.array([
        [0],
        [3],
        [1],
        [2],
    ]))


def test_flat_index_loc_grouper():
    matrix = np.array([
        [[0, 1, 0],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 0, 1]],
        [[0, 0, 1],
         [0, 1, 1]],
        [[0, 0, 1],
         [0, 0, 2]],
    ])
    assert len(np.unique(np.sum(flatten_matrix(matrix), axis=1))) == 1

    loc_grouper = FlatIndexLocationGrouper()
    encoder = AmountFirstGroupedEncoder(FirstImputer(), TotalAmountGrouper(), loc_grouper)
    encoder.matrix = matrix

    assert len(encoder.design_vars) == 3
    assert np.all(loc_grouper._prepared_dvs == np.array([
        [1, 3, 4],
        [2, 3, 4],
        [2, 3, 5],
        [2, 4, 5],
        [2, 5, 5],
    ]))
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 2, 0],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 1],
        [1, 2, 1],
    ]))


def test_rel_flat_index_loc_grouper():
    matrix = np.array([
        [[0, 1, 0],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 0, 1]],
        [[0, 0, 1],
         [0, 1, 1]],
        [[0, 0, 1],
         [0, 0, 2]],
    ])
    assert len(np.unique(np.sum(flatten_matrix(matrix), axis=1))) == 1

    loc_grouper = RelFlatIndexLocationGrouper()
    encoder = AmountFirstGroupedEncoder(FirstImputer(), TotalAmountGrouper(), loc_grouper)
    encoder.matrix = matrix

    assert len(encoder.design_vars) == 3
    assert np.all(loc_grouper._prepared_dvs == np.array([
        [1, 2, 1],
        [2, 1, 1],
        [2, 1, 2],
        [2, 2, 1],
        [2, 3, 0],
    ]))
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 2, 0],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(encoder._design_vectors == np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 0, 2],
        [1, 1, 1],
        [1, 2, 0],
    ]))


def test_coord_index_loc_grouper():
    matrix = np.array([
        [[0, 1, 0],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 0, 1]],
        [[0, 0, 1],
         [0, 1, 1]],
        [[0, 0, 1],
         [0, 0, 2]],
    ])
    assert len(np.unique(np.sum(flatten_matrix(matrix), axis=1))) == 1

    loc_grouper = CoordIndexLocationGrouper()
    encoder = AmountFirstGroupedEncoder(FirstImputer(), TotalAmountGrouper(), loc_grouper)
    encoder.matrix = matrix

    assert len(encoder.design_vars) == 3
    assert np.all(loc_grouper._prepared_dvs == np.array([
        [0, 1, 1, 0, 1, 1],
        [0, 2, 1, 0, 1, 1],
        [0, 2, 1, 0, 1, 2],
        [0, 2, 1, 1, 1, 2],
        [0, 2, 1, 2, 1, 2],
    ]))
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 2, 0],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 1],
        [1, 2, 1],
    ]))


def test_rel_coord_index_loc_grouper():
    matrix = np.array([
        [[0, 1, 0],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 1, 0]],
        [[0, 0, 1],
         [1, 0, 1]],
        [[0, 0, 1],
         [0, 1, 1]],
        [[0, 0, 1],
         [0, 0, 2]],
    ])
    assert len(np.unique(np.sum(flatten_matrix(matrix), axis=1))) == 1

    loc_grouper = RelCoordIndexLocationGrouper()
    encoder = AmountFirstGroupedEncoder(FirstImputer(), TotalAmountGrouper(), loc_grouper)
    encoder.matrix = matrix

    assert len(encoder.design_vars) == 3
    assert np.all(loc_grouper._prepared_dvs == np.array([
        [0, 1, 1, -1, 0, 1],
        [0, 2, 1, -2, 0, 1],
        [0, 2, 1, -2, 0, 2],
        [0, 2, 1, -1, 0, 1],
        [0, 2, 1,  0, 0, 0],
    ]))
    assert np.all(encoder._design_vectors == np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 2, 0],
    ]))

    encoder.normalize_within_group = False
    encoder.matrix = matrix
    assert np.all(encoder._design_vectors == np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 0, 2],
        [1, 1, 1],
        [1, 2, 0],
    ]))