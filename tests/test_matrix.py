import numpy as np
import pytest
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
    assert set(AggregateAssignmentMatrixGenerator(src=[nodes[0]], tgt=[nodes[1]])._iter_sources()) == {
        (1,),
    }
    assert set(AggregateAssignmentMatrixGenerator(src=[inf_nodes[0]], tgt=[nodes[1]])._iter_sources()) == {
        (1,),
    }

    obj = Node([1, 2])
    assert set(AggregateAssignmentMatrixGenerator(src=[obj], tgt=[nodes[1]])._iter_sources()) == {
        (1,),
    }

    obj2 = Node([0, 1, 2])
    assert set(AggregateAssignmentMatrixGenerator(src=[obj, obj2], tgt=[nodes[1]])._iter_sources()) == {
        (1, 0), (1, 1),
    }

    assert set(AggregateAssignmentMatrixGenerator(src=[obj, inf_nodes[0]], tgt=[nodes[1]])._iter_sources()) == {
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
    assert np.all(gen._agg_matrices(matrix_gen) == assert_matrix)


def test_iter_permuted_conns(nodes):
    gen = AggregateAssignmentMatrixGenerator(src=[Node([0, 1]), Node([0, 1])], tgt=[nodes[2]])
    _assert_matrix(gen, gen._iter_matrices([1, 0], [1]), np.array([
        [[1], [0]],
    ]))

    gen = AggregateAssignmentMatrixGenerator(src=[nodes[0], nodes[1]], tgt=[nodes[2], nodes[3]])
    _assert_matrix(gen, gen._iter_matrices([1, 1], [1, 1]), np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ]))
    _assert_matrix(gen, gen._iter_matrices((1, 1), (1, 1)), np.array([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ]))

    gen.ex = [(gen.src[1], gen.tgt[0])]
    _assert_matrix(gen, gen._iter_matrices([1, 1], [1, 1]), np.array([
        [[1, 0], [0, 1]],
    ]))
    gen.ex = None

    for _ in range(20):
        _assert_matrix(gen, gen._iter_matrices([1, 2], [1, 2]), np.array([
            [[1, 0], [0, 2]],
            [[0, 1], [1, 1]],
        ]))


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
        assert set(gen.iter_conns()) == {
            ((obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj2, obj4)),
            ((obj1, obj4), (obj2, obj3)),
            ((obj1, obj4), (obj1, obj4), (obj2, obj4)),
            ((obj1, obj3), (obj1, obj4), (obj2, obj4)),
            ((obj1, obj4), (obj1, obj4), (obj2, obj3)),
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
    matrix = gen.get_agg_matrix()
    emtpy_matrix = matrix[:0, :, :]

    assert np.all(gen.filter_matrices(matrix, [True, True], [True, True]) == matrix)
    assert np.all(gen.filter_matrices(matrix, [True, False], [True, True]) == emtpy_matrix)

    assert np.all(gen.filter_matrices(matrix, [True, True], [False, True]) == np.array([
        [[0, 1], [0, 1]],
        [[0, 2], [0, 1]],
    ]))
