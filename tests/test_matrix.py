import pytest
import numpy as np
from assign_enc.matrix import *


@pytest.fixture
def nodes():
    return [Node([1]) for _ in range(10)]


@pytest.fixture
def inf_nodes():
    return [Node(min_conn=1) for _ in range(10)]


def test_conn_override_map(nodes, inf_nodes):
    assert AggregateAssignmentMatrixGenerator(src=[nodes[0]], tgt=[nodes[1]])._get_max_conn() == 1

    assert AggregateAssignmentMatrixGenerator(src=[nodes[0], inf_nodes[0]], tgt=[nodes[1]])._get_max_conn() == 2
    assert AggregateAssignmentMatrixGenerator(src=[nodes[0], inf_nodes[0]],
                                              tgt=[nodes[1], inf_nodes[1]])._get_max_conn() == 2

    obj_spec_00 = Node(min_conn=0)
    obj_spec_01 = Node(min_conn=0)
    assert AggregateAssignmentMatrixGenerator(src=[nodes[0], obj_spec_00],
                                              tgt=[nodes[1], obj_spec_01])._get_max_conn() == 2

    obj_spec_00 = Node(min_conn=2)
    obj_spec_01 = Node(min_conn=0)
    assert AggregateAssignmentMatrixGenerator(src=[nodes[0], obj_spec_00],
                                              tgt=[nodes[1], obj_spec_01])._get_max_conn() == 3


def test_iter_conns(nodes, inf_nodes):
    assert set(AggregateAssignmentMatrixGenerator(src=[nodes[0]], tgt=[nodes[1]]).iter_sources()) == {
        (1,),
    }
    assert set(AggregateAssignmentMatrixGenerator(src=[inf_nodes[0]], tgt=[nodes[1]]).iter_sources()) == {
        (1,),
    }

    obj = Node([1, 2])
    assert set(AggregateAssignmentMatrixGenerator(src=[obj], tgt=[nodes[1]]).iter_sources()) == {
        (1,),
    }

    obj2 = Node([0, 1, 2])
    assert set(AggregateAssignmentMatrixGenerator(src=[obj, obj2], tgt=[nodes[1]]).iter_sources()) == {
        (1, 0), (1, 1),
    }

    assert set(AggregateAssignmentMatrixGenerator(src=[obj, inf_nodes[0]], tgt=[nodes[1]]).iter_sources()) == {
        (1, 1),
    }

    obj2 = Node([0, 1, 2])
    assert set(AggregateAssignmentMatrixGenerator(src=[obj, obj2], tgt=[nodes[1]])._iter_conn_slots([obj, obj2], n=2)) == {
        (1, 1),
    }

    assert set(AggregateAssignmentMatrixGenerator(src=[obj, obj2], tgt=[nodes[1]])._iter_conn_slots([obj, obj2])) == {
        (1, 0), (1, 1),
    }
    assert set(AggregateAssignmentMatrixGenerator(src=[obj, obj2], tgt=[Node([0, 1, 2])])._iter_conn_slots([obj, obj2])) == {
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2),
    }


def test_iter_order():
    for _ in range(100):
        node1 = Node([1])
        nodes = [Node([0, 1]) for _ in range(5)]
        assert list(AggregateAssignmentMatrixGenerator(src=[node1], tgt=nodes)._iter_conn_slots(nodes, n=1)) == \
               [tuple(1 if j == i else 0 for j in range(len(nodes))) for i in reversed(range(len(nodes)))]


def _assert_matrix(gen: AggregateAssignmentMatrixGenerator, matrix_gen, assert_matrix: np.ndarray):
    agg_matrix = gen._agg_matrices(matrix_gen)
    assert agg_matrix.shape == assert_matrix.shape
    assert np.all(agg_matrix == assert_matrix)


def test_iter_permuted_conns(nodes):
    gen = AggregateAssignmentMatrixGenerator(src=[Node([0, 1]), Node([0, 1])], tgt=[nodes[2]])
    _assert_matrix(gen, gen._iter_matrices([1, 0], [1]), np.array([
        [[1], [0]],
    ]))
    assert gen.count_matrices([1, 0], [1]) == 1

    gen = AggregateAssignmentMatrixGenerator(src=[nodes[0], nodes[1]], tgt=[nodes[2], nodes[3]])
    _assert_matrix(gen, gen._iter_matrices([1, 1], [1, 1]), np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ]))
    assert gen.count_matrices([1, 1], [1, 1]) == 2
    _assert_matrix(gen, gen._iter_matrices((1, 1), (1, 1)), np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ]))
    assert gen.validate_matrix(np.array([[1, 0], [0, 1]]))
    assert not gen.validate_matrix(np.array([[1, 0], [1, 0]]))
    assert not gen.validate_matrix(np.array([[1, 1], [0, 0]]))

    gen.ex = [(gen.src[1], gen.tgt[0])]
    _assert_matrix(gen, gen._iter_matrices([1, 1], [1, 1]), np.array([
        [[1, 0], [0, 1]],
    ]))
    assert gen.count_matrices([1, 1], [1, 1]) == 1
    gen.ex = None

    for _ in range(20):
        _assert_matrix(gen, gen._iter_matrices([1, 2], [1, 2]), np.array([
            [[1, 0], [0, 2]],
            [[0, 1], [1, 1]],
        ]))
        assert gen.count_matrices([1, 2], [1, 2]) == 2


def test_iter_matrix_zero_conn():
    gen = AggregateAssignmentMatrixGenerator(src=[Node([0, 1]), Node([0, 1])], tgt=[Node([0, 1])])
    _assert_matrix(gen, gen._iter_matrices([1, 0], [1]), np.array([
        [[1], [0]],
    ]))
    assert gen.count_matrices([1, 0], [1]) == 1
    _assert_matrix(gen, gen._iter_matrices([0, 0], [0]), np.array([
        [[0], [0]],
    ]))
    assert gen.count_matrices([0, 0], [0]) == 1


def test_iter_edges():
    obj1 = Node(nr_conn_list=[1, 2])
    obj2 = Node(nr_conn_list=[1])
    obj3 = Node(nr_conn_list=[0, 1])
    obj4 = Node(min_conn=1)
    gen = AggregateAssignmentMatrixGenerator(src=[obj1, obj2], tgt=[obj3, obj4])

    for _ in range(10):
        _assert_matrix(gen, gen, np.array([
            [[0, 1], [0, 1]],
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[0, 2], [0, 1]],
            [[1, 1], [0, 1]],
            [[0, 2], [1, 0]],
        ]))
        assert gen.count_all_matrices() == 6
        assert set(gen.iter_conns()) == {
            ((obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj2, obj4)),
            ((obj1, obj4), (obj2, obj3)),
            ((obj1, obj4), (obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj1, obj4), (obj2, obj4)),
            ((obj1, obj4), (obj1, obj4), (obj2, obj3)),
        }

    obj1.rep = False
    gen._no_repeat_mat = None
    for _ in range(10):
        assert set(gen.iter_conns()) == {
            ((obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj2, obj4)),
            ((obj1, obj4), (obj2, obj3)),
            ((obj1, obj3), (obj1, obj4), (obj2, obj4)),
        }

    gen.ex = [(obj2, obj3)]
    for _ in range(10):
        assert set(gen.iter_conns()) == {
            ((obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj2, obj4)),
            ((obj1, obj3), (obj1, obj4), (obj2, obj4)),
        }


def test_iter_edges_no_repeat():
    obj1 = Node(nr_conn_list=[1, 2])
    obj2 = Node(nr_conn_list=[1])
    obj3 = Node(nr_conn_list=[0, 1])
    obj4 = Node(min_conn=1, repeated_allowed=False)
    gen = AggregateAssignmentMatrixGenerator(src=[obj1, obj2], tgt=[obj3, obj4])

    for _ in range(10):
        assert set(gen.iter_conns()) == {
            ((obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj2, obj4)),
            ((obj1, obj4), (obj2, obj3)),
            # ((obj1, obj4), (obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj1, obj4), (obj2, obj4)),
            # ((obj1, obj4), (obj1, obj4), (obj2, obj3)),
        }

    obj1.rep = False
    for _ in range(10):
        assert set(gen.iter_conns()) == {
            ((obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj2, obj4)),
            ((obj1, obj4), (obj2, obj3)),
            ((obj1, obj3), (obj1, obj4), (obj2, obj4)),
        }

    gen.ex = [(obj2, obj3)]
    for _ in range(10):
        assert set(gen.iter_conns()) == {
            ((obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj2, obj4)),
            ((obj1, obj3), (obj1, obj4), (obj2, obj4)),
        }


def test_filter_matrices():
    obj1 = Node(nr_conn_list=[1, 2])
    obj2 = Node(nr_conn_list=[1])
    obj3 = Node(nr_conn_list=[0, 1])
    obj4 = Node(min_conn=1)
    gen = AggregateAssignmentMatrixGenerator(src=[obj1, obj2], tgt=[obj3, obj4])

    _assert_matrix(gen, gen, np.array([
        [[0, 1], [0, 1]],
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[0, 2], [0, 1]],
        [[1, 1], [0, 1]],
        [[0, 2], [1, 0]],
    ]))
    assert gen.count_all_matrices() == 6
    matrix = gen.get_agg_matrix()
    all_mask = np.ones((matrix.shape[0],), dtype=bool)
    none_mask = np.zeros((matrix.shape[0],), dtype=bool)
    for i in range(matrix.shape[0]):
        assert gen.validate_matrix(matrix[i, :, :])

    assert np.all(gen.filter_matrices(matrix, [True, True], [True, True]) == all_mask)
    assert np.all(gen.filter_matrices(matrix) == all_mask)
    assert np.all(gen.filter_matrices(matrix, tgt_exists=[True, True]) == all_mask)
    assert np.all(gen.filter_matrices(matrix, [True, False], [True, True]) == none_mask)
    assert np.all(gen.filter_matrices(matrix, src_exists=[True, False]) == none_mask)

    assert np.all(gen.filter_matrices(matrix, [True, True], [False, True]) ==
                  np.array([True, False, False, True, False, False], dtype=bool))
    assert np.all(gen.filter_matrices(matrix, tgt_exists=[False, True]) ==
                  np.array([True, False, False, True, False, False], dtype=bool))

    assert gen.validate_matrix(matrix[0, :, :], tgt_exists=[False, True])
    assert not gen.validate_matrix(matrix[1, :, :], tgt_exists=[False, True])

    matrix2 = gen.get_agg_matrix(cache=True)
    assert np.all(matrix == matrix2)


def test_matrix_all_inf():
    gen = AggregateAssignmentMatrixGenerator(src=[Node(min_conn=0), Node(min_conn=0)],
                                             tgt=[Node(min_conn=0), Node(min_conn=0)])

    _assert_matrix(gen, gen, np.array([  # 26 diff
        [[0, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 2]],
        [[0, 0], [1, 1]],
        [[0, 0], [2, 0]],

        [[0, 1], [0, 0]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, 1]],
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[1, 0], [1, 0]],
        [[1, 0], [0, 2]],
        [[0, 1], [1, 1]],
        [[1, 0], [1, 1]],
        [[0, 1], [2, 0]],

        [[0, 2], [0, 0]],
        [[1, 1], [0, 0]],
        [[2, 0], [0, 0]],
        [[1, 1], [0, 1]],
        [[0, 2], [1, 0]],
        [[2, 0], [0, 1]],
        [[1, 1], [1, 0]],
        [[2, 0], [0, 2]],
        [[1, 1], [1, 1]],
        [[0, 2], [2, 0]],
    ]))
    assert gen.count_all_matrices() == 26

    for matrix in gen:
        assert gen.validate_matrix(matrix)


def test_matrix_all_inf_no_repeat():
    gen = AggregateAssignmentMatrixGenerator(
        src=[Node(min_conn=0, repeated_allowed=False), Node(min_conn=0, repeated_allowed=False)],
        tgt=[Node(min_conn=0, repeated_allowed=False), Node(min_conn=0, repeated_allowed=False)],
    )

    _assert_matrix(gen, gen, np.array([  # 16 diff
        [[0, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[0, 0], [1, 0]],
        [[0, 0], [1, 1]],

        [[0, 1], [0, 0]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, 1]],
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[1, 0], [1, 0]],
        [[0, 1], [1, 1]],
        [[1, 0], [1, 1]],

        [[1, 1], [0, 0]],
        [[1, 1], [0, 1]],
        [[1, 1], [1, 0]],
        [[1, 1], [1, 1]],
    ]))
    assert gen.count_all_matrices() == 16

    for matrix in gen:
        assert gen.validate_matrix(matrix)


def test_matrix_all_inf_no_repeat_23():
    gen = AggregateAssignmentMatrixGenerator(src=[Node(min_conn=0, repeated_allowed=False) for _ in range(2)],
                                             tgt=[Node(min_conn=0, repeated_allowed=False) for _ in range(3)])
    assert gen.max_conn == 3
    assert list(gen._iter_conn_slots(gen.src)) == [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 0), (3, 1), (3, 2), (3, 3),
    ]

    assert list(gen._iter_conn_slots(gen.tgt, is_src=False, n=2)) == [
        (0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0),
    ]

    _assert_matrix(gen, gen._iter_matrices([0, 2], [0, 0, 2]), np.empty((0, 2, 3)))
    _assert_matrix(gen, gen._iter_matrices([0, 2], [0, 2, 0]), np.empty((0, 2, 3)))
    _assert_matrix(gen, gen._iter_matrices([0, 2], [2, 0, 0]), np.empty((0, 2, 3)))
    assert gen.count_matrices([0, 2], [2, 0, 0]) == 0
    _assert_matrix(gen, gen._iter_matrices([0, 2], [0, 1, 1]), np.array([[[0, 0, 0], [0, 1, 1]]]))
    _assert_matrix(gen, gen._iter_matrices([0, 2], [1, 1, 0]), np.array([[[0, 0, 0], [1, 1, 0]]]))
    _assert_matrix(gen, gen._iter_matrices([0, 2], [1, 0, 1]), np.array([[[0, 0, 0], [1, 0, 1]]]))
    assert gen.count_matrices([0, 2], [1, 1, 0]) == 1

    assert list(gen._iter_conn_slots(gen.tgt, is_src=False, n=3)) == [
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 1, 1), (1, 2, 0), (2, 0, 1), (2, 1, 0),
    ]
    _assert_matrix(gen, gen._iter_matrices([0, 3], [0, 1, 2]), np.empty((0, 2, 3)))
    _assert_matrix(gen, gen._iter_matrices([0, 3], [1, 1, 1]), np.array([[[0, 0, 0], [1, 1, 1]]]))
    assert gen.count_matrices([0, 3], [1, 1, 1]) == 1

    gen0 = AggregateAssignmentMatrixGenerator(src=[Node([0]), Node(min_conn=0, repeated_allowed=False)],
                                              tgt=[Node(min_conn=0, repeated_allowed=False) for _ in range(3)])
    assert gen0.max_conn == 3
    assert list(gen0._iter_conn_slots(gen0.src)) == [
        (0, 0), (0, 1), (0, 2), (0, 3),
    ]
    _assert_matrix(gen0, gen0._iter_matrices((0, 2), (1, 0, 1)), np.array([[[0, 0, 0], [1, 0, 1]]]))

    _assert_matrix(gen0, gen0, np.array([
        [[0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 1, 1]],
        [[0, 0, 0], [1, 0, 1]],
        [[0, 0, 0], [1, 1, 0]],
        [[0, 0, 0], [1, 1, 1]],
    ]))
    assert gen0.count_all_matrices() == 8


def test_count_n_pool_take():
    def _count_no_dup(n_p, n_t):
        return count_n_pool_take(n_p, n_t, (0,)*n_p)[0]

    def _count_dup(n_p, n_t, n_dup):
        return count_n_pool_take(n_p, n_t, n_dup)[0]

    for _ in range(1000):
        assert _count_no_dup(0, 0) == 1
        assert _count_no_dup(0, 1) == 0
        assert _count_no_dup(1, 0) == 1
        assert _count_no_dup(1, 1) == 1

        assert [_count_no_dup(2, n) for n in range(3)] == [1, 2, 1]
        assert [_count_no_dup(3, n) for n in range(4)] == [1, 3, 3, 1]
        assert [_count_no_dup(4, n) for n in range(5)] == [1, 4, 6, 4, 1]
        assert [_count_no_dup(5, n) for n in range(6)] == [1, 5, 10, 10, 5, 1]
        assert [_count_no_dup(6, n) for n in range(7)] == [1, 6, 15, 20, 15, 6, 1]

        assert _count_dup(2, 1, (1, 0)) == 1
        assert _count_dup(2, 2, (1, 0)) == 1
        assert _count_dup(3, 1, (1, 0, 0)) == 2
        assert _count_dup(3, 1, (0, 1, 0)) == 2
        assert _count_dup(3, 1, (2, 0, 0)) == 1
        assert _count_dup(3, 2, (1, 0, 0)) == 2
        assert _count_dup(3, 2, (0, 1, 0)) == 2
        assert _count_dup(3, 2, (2, 0, 0)) == 1
        assert _count_dup(3, 3, (0, 1, 0)) == 1

        assert _count_dup(4, 1, (1, 0, 0, 0)) == 3
        assert _count_dup(4, 1, (0, 1, 0, 0)) == 3
        assert _count_dup(4, 1, (0, 0, 1, 0)) == 3
        assert _count_dup(4, 1, (2, 0, 0, 0)) == 2
        assert _count_dup(4, 1, (0, 2, 0, 0)) == 2
        assert _count_dup(4, 1, (3, 0, 0, 0)) == 1
        assert _count_dup(4, 2, (1, 0, 0, 0)) == 4
        assert _count_dup(4, 2, (0, 1, 0, 0)) == 4
        assert _count_dup(4, 2, (0, 0, 1, 0)) == 4
        assert _count_dup(4, 2, (2, 0, 0, 0)) == 2
        assert _count_dup(4, 2, (0, 2, 0, 0)) == 2
        assert _count_dup(4, 2, (3, 0, 0, 0)) == 1
        assert _count_dup(4, 3, (1, 0, 0, 0)) == 3
        assert _count_dup(4, 3, (0, 1, 0, 0)) == 3
        assert _count_dup(4, 3, (0, 0, 1, 0)) == 3
        assert _count_dup(4, 3, (2, 0, 0, 0)) == 2
        assert _count_dup(4, 3, (0, 2, 0, 0)) == 2
        assert _count_dup(4, 3, (3, 0, 0, 0)) == 1
        assert _count_dup(4, 4, (1, 0, 0, 0)) == 1

        assert _count_dup(6, 3, (1, 0, 0, 2, 0, 0)) == 6


def test_count_n_pool_combs():
    def _assert_count_no_dup(n_p, n_t, combinations):
        n, combs = count_n_pool_take(n_p, n_t, (0,)*n_p)
        assert combinations.shape == (n, n_p)
        assert np.all(combs == combinations)

    def _assert_count_dup(n_p, n_t, n_dup, combinations):
        n, combs = count_n_pool_take(n_p, n_t, n_dup)
        assert combinations.shape == (n, n_p)
        assert np.all(combs == combinations)

    _assert_count_no_dup(0, 0, np.zeros((1, 0), dtype=bool))
    _assert_count_no_dup(1, 0, np.array([[0]]))
    _assert_count_no_dup(1, 1, np.array([[1]]))

    _assert_count_no_dup(3, 3, np.array([[1, 1, 1]]))
    _assert_count_no_dup(3, 2, np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]))
    _assert_count_no_dup(5, 3, np.array([
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 1, 1, 1],
    ]))

    _assert_count_dup(2, 1, (1, 0), np.array([[1, 0]]))
    _assert_count_dup(2, 2, (1, 0), np.array([[1, 1]]))
    _assert_count_dup(3, 1, (1, 0, 0), np.array([[1, 0, 0], [0, 0, 1]]))
    _assert_count_dup(4, 1, (0, 1, 0, 0), np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]))
    _assert_count_dup(4, 2, (0, 2, 0, 0), np.array([[1, 1, 0, 0], [0, 1, 1, 0]]))

    _assert_count_dup(6, 2, (1, 0, 0, 2, 0, 0), np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
    ]))
    _assert_count_dup(6, 3, (1, 0, 0, 2, 0, 0), np.array([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1],
    ]))
    _assert_count_dup(6, 4, (1, 0, 0, 2, 0, 0), np.array([
        [1, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 1],
        [0, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
    ]))


def test_count_matrices_from_nodes():
    src = [Node(min_conn=0, repeated_allowed=False) for _ in range(2)]
    tgt = [Node(min_conn=0, repeated_allowed=False) for _ in range(3)]
    gen = AggregateAssignmentMatrixGenerator(src, tgt)

    assert gen.count_matrices((0, 3), (0, 1, 2)) == 0

    assert gen.count_matrices((1, 1), (1, 1, 0)) == 2
    assert len(list(gen._iter_matrices((1, 1), (1, 1, 0)))) == 2

    assert gen.count_matrices((2, 1), (1, 1, 1)) == 3
    assert len(list(gen._iter_matrices((2, 1), (1, 1, 1)))) == 3

    for src_nr in gen.iter_sources():  # Iterate over source node nrs
        for tgt_nr in gen.iter_targets(n_source=sum(src_nr)):  # Iterate over target node nrs
            n_mat = gen.count_matrices(src_nr, tgt_nr)
            matrices = np.array(list(gen._iter_matrices(src_nr, tgt_nr)))
            assert matrices.shape[0] == n_mat

    matrix = gen.get_agg_matrix()
    assert matrix.shape[0] == 64
    assert gen.count_all_matrices() == 64
