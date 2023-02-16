import timeit
from assign_enc.matrix import *
from assign_enc.selector import *
from assign_enc.assignment_manager import *


def test_selector():
    src = [Node([1], repeated_allowed=False) for _ in range(2)]
    tgt = [Node([0, 1], repeated_allowed=False) for _ in range(4)]
    selector = EncoderSelector(MatrixGenSettings(src, tgt))

    assert selector._get_n_mat() == (12, 1)

    assignment_manager = selector._get_best_assignment_manager()
    assert isinstance(assignment_manager, AssignmentManagerBase)

    encoder = assignment_manager.encoder
    assert encoder.get_imputation_ratio() < 10


def test_selector_inf_idx_filter():
    src = [Node([1], repeated_allowed=False) for _ in range(2)]
    tgt = [Node([0, 1], repeated_allowed=False) for _ in range(4)]
    selector = EncoderSelector(MatrixGenSettings(src, tgt))
    selector.n_mat_max_eager = 1
    selector.min_distance_correlation = 1.2
    assert selector._get_n_mat() == (12, 1)

    assignment_manager = selector._get_best_assignment_manager()
    assert isinstance(assignment_manager, AssignmentManagerBase)

    encoder = assignment_manager.encoder
    assert encoder.get_imputation_ratio() < 10


def test_selector_encoding_timeout():
    src = [Node([1], repeated_allowed=False) for _ in range(6)]
    tgt = [Node([0, 1], repeated_allowed=False) for _ in range(6)]
    selector = EncoderSelector(MatrixGenSettings(src, tgt))
    selector.encoding_timeout = .05

    assignment_manager = selector._get_best_assignment_manager()
    assert isinstance(assignment_manager, AssignmentManagerBase)

    encoder = assignment_manager.encoder
    assert encoder.get_imputation_ratio() == 1


def test_selector_cache():
    src = [Node([1], repeated_allowed=False) for _ in range(4)]
    tgt = [Node([0, 1], repeated_allowed=False) for _ in range(4)]
    selector = EncoderSelector(MatrixGenSettings(src, tgt))

    selector.reset_cache()

    s = timeit.default_timer()
    selector.get_best_assignment_manager()
    t1 = timeit.default_timer()-s

    s = timeit.default_timer()
    selector.get_best_assignment_manager()
    t2 = timeit.default_timer()-s
    assert t2 < t1

    s = timeit.default_timer()
    selector.get_best_assignment_manager(cache=False)
    t3 = timeit.default_timer()-s
    assert t3 > t2


def test_one_to_one(gen_one_per_existence: AggregateAssignmentMatrixGenerator):
    g = gen_one_per_existence
    selector = EncoderSelector(g.settings)

    assignment_manager = selector._get_best_assignment_manager()
    assert isinstance(assignment_manager, AssignmentManagerBase)

    assert len(assignment_manager.design_vars) == 0


def test_selector_no_exist():
    src = [Node([1], repeated_allowed=False)]
    tgt = [Node([2], repeated_allowed=False)]
    selector = EncoderSelector(MatrixGenSettings(src, tgt))
    assert isinstance(selector._get_best_assignment_manager(), AssignmentManagerBase)
