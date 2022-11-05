import numpy as np
from assign_enc.matrix import *
from assign_enc.lazy.encodings.group_amount import *
from assign_enc.lazy.imputation.constraint_violation import *


def test_flat_amount_encoder():
    dvs = FlatLazyAmountEncoder().encode(2, 2, [((0, 1), (0, 1)), ((1, 1), (0, 2)), ((1, 1), (2, 0)), ((1, 2), (2, 1))],
                                         NodeExistence())
    assert len(dvs) == 1
    assert dvs[0].n_opts == 4


def test_flat_connection_encoder():
    encoder = LazyAmountFirstEncoder(LazyConstraintViolationImputer(), FlatLazyAmountEncoder(), FlatLazyConnectionEncoder())
    encoder.set_nodes(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)])

    n_tgt_n_src = list(encoder.iter_n_src_n_tgt())
    assert len(n_tgt_n_src) == 16
    assert ((2, 2), (1, 3), NodeExistence()) in n_tgt_n_src

    assert len(encoder.design_vars) == 2
    assert encoder.design_vars[0].n_opts == 16

    matrices_max = encoder.get_matrices((2, 2), (1, 3))
    assert matrices_max.shape[0] == 2
    assert encoder.design_vars[1].n_opts == 2

    dv, matrix = encoder.get_matrix([15, 0])
    assert np.all(dv == [15, 0])
    assert np.all(matrix == matrices_max[0, :, :])
    assert np.all(matrix == np.array([[1, 1], [0, 2]]))
    dv, matrix = encoder.get_matrix([15, 1])
    assert np.all(dv == [15, 1])
    assert np.all(matrix == matrices_max[1, :, :])

    assert encoder.get_n_design_points() == 32
    # assert encoder._matrix_gen.count_all_matrices() == 21
    assert encoder.get_imputation_ratio() == 32/21


def test_flat_connection_encoder_multi_max():
    encoder = LazyAmountFirstEncoder(LazyConstraintViolationImputer(), FlatLazyAmountEncoder(), FlatLazyConnectionEncoder())
    encoder.set_nodes(
        src=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)],
        tgt=[Node([1], repeated_allowed=False) for _ in range(3)],
    )

    assert len(list(encoder.iter_n_src_n_tgt())) == 4
    assert encoder.matrix_gen.count_all_matrices() == 8

    assert len(encoder.design_vars) == 2
    assert encoder.design_vars[0].n_opts == 4
    assert encoder.design_vars[1].n_opts == 3


def test_total_amount_encoder():
    dvs = TotalLazyAmountEncoder().encode(2, 2, [((0, 1), (0, 1)), ((1, 1), (0, 2)), ((1, 1), (2, 0)), ((1, 2), (2, 1))],
                                          NodeExistence())
    assert len(dvs) == 2
    assert dvs[0].n_opts == 3
    assert dvs[1].n_opts == 2

    dvs = TotalLazyAmountEncoder().encode(2, 2, [((0, 1), (0, 1)), ((1, 1), (0, 2))], NodeExistence())
    assert len(dvs) == 1
    assert dvs[0].n_opts == 2

    encoder = LazyAmountFirstEncoder(LazyConstraintViolationImputer(), TotalLazyAmountEncoder(), FlatLazyConnectionEncoder())
    encoder.set_nodes(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)])

    n_tgt_n_src = list(encoder.iter_n_src_n_tgt())
    assert len(n_tgt_n_src) == 16
    assert len(encoder.design_vars) == 3
    assert encoder.design_vars[0].n_opts == 4
    assert encoder.design_vars[1].n_opts == 6

    dv, matrix = encoder.get_matrix([0, 5, 0])
    assert np.all(dv == [0, 5, 0])
    assert np.all(matrix == -1)

    dv, matrix = encoder.get_matrix([3, 1, 0])
    assert np.all(matrix == np.array([[1, 1], [0, 2]]))


def test_source_amount_encoder():
    dvs = SourceLazyAmountEncoder().encode(2, 2, [((0, 1), (0, 1)), ((1, 1), (0, 2)), ((1, 1), (2, 0)), ((1, 2), (2, 1))],
                                           NodeExistence())
    assert len(dvs) == 3
    assert dvs[0].n_opts == 2
    assert dvs[1].n_opts == 2
    assert dvs[2].n_opts == 2

    encoder = LazyAmountFirstEncoder(LazyConstraintViolationImputer(), SourceLazyAmountEncoder(), FlatLazyConnectionEncoder())
    encoder.set_nodes(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)])

    n_tgt_n_src = list(encoder.iter_n_src_n_tgt())
    assert len(n_tgt_n_src) == 16
    assert len(encoder.design_vars) == 4
    assert encoder.design_vars[0].n_opts == 3
    assert encoder.design_vars[1].n_opts == 4
    assert encoder.design_vars[2].n_opts == 2

    dv, matrix = encoder.get_matrix([0, 0, 0, 0])
    assert np.all(dv == [0, 0, 0, 0])
    assert np.all(matrix == np.array([[0, 0], [0, 1]]))


def test_source_target_amount_encoder():
    dvs = SourceTargetLazyAmountEncoder().encode(
        2, 2, [((0, 1), (0, 1)), ((1, 1), (0, 2)), ((1, 1), (2, 0)), ((1, 2), (2, 1))], NodeExistence())
    assert len(dvs) == 3
    assert dvs[0].n_opts == 2
    assert dvs[1].n_opts == 2
    assert dvs[2].n_opts == 2

    encoder = LazyAmountFirstEncoder(LazyConstraintViolationImputer(), SourceTargetLazyAmountEncoder(), FlatLazyConnectionEncoder())
    encoder.set_nodes(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)])

    n_tgt_n_src = list(encoder.iter_n_src_n_tgt())
    assert len(n_tgt_n_src) == 16
    assert len(encoder.design_vars) == 4
    assert encoder.design_vars[0].n_opts == 3
    assert encoder.design_vars[1].n_opts == 4
    assert encoder.design_vars[2].n_opts == 2


def test_filter_dvs():
    encoder = LazyAmountFirstEncoder(LazyConstraintViolationImputer(), FlatLazyAmountEncoder(), FlatLazyConnectionEncoder())
    encoder.set_nodes(src=[Node([1], repeated_allowed=False), Node([1], repeated_allowed=False)],
                      tgt=[Node([1], repeated_allowed=False), Node([1], repeated_allowed=False)])

    n_tgt_n_src = list(encoder.iter_n_src_n_tgt())
    assert len(n_tgt_n_src) == 1

    assert len(encoder.design_vars) == 1
    assert encoder.design_vars[0].n_opts == 2

    dv, matrix = encoder.get_matrix([0])
    assert np.all(dv == [0])
    assert np.all(matrix == np.array([[1, 0], [0, 1]]))

    dv, matrix = encoder.get_matrix([1])
    assert np.all(dv == [1])
    assert np.all(matrix == np.array([[0, 1], [1, 0]]))
