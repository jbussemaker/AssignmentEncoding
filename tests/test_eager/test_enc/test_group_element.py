import numpy as np
from assign_enc.matrix import *
from assign_enc.eager.imputation.first import *
from assign_enc.eager.imputation.closest import *
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


def test_encoder():
    # Combinations with replacement
    gen = AggregateAssignmentMatrixGenerator(src=[Node([3])], tgt=[Node(min_conn=0) for _ in range(3)])

    matrix = gen.get_agg_matrix()[NodeExistence()]
    encoder = ElementGroupedEncoder(ClosestImputer(), matrix)
    assert encoder.n_mat_max == 10

    assert len(encoder.design_vars) == 2
    assert encoder.design_vars[0].n_opts == 4
    assert encoder.design_vars[1].n_opts == 4

    dv_seen = set()
    for i0 in range(encoder.design_vars[0].n_opts):
        for i1 in range(encoder.design_vars[1].n_opts):
            dv, mat = encoder.get_matrix([i0, i1])
            assert encoder.is_valid_vector(dv)
            assert gen.validate_matrix(mat)
            if not np.all(dv == np.array([i0, i1])):
                assert not encoder.is_valid_vector([i0, i1])
            dv_seen.add(tuple(dv))

    assert len(dv_seen) == 10


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    encoder = ElementGroupedEncoder(ClosestImputer())
    encoder.matrix = gen_one_per_existence.get_agg_matrix()
    assert len(encoder.design_vars) == 0

    assert encoder.get_n_design_points() == 1
    assert encoder.get_imputation_ratio() == 1.2
    assert encoder.get_information_index() == 1
    assert encoder.get_distance_correlation() == 1
