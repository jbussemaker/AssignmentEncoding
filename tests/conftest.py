import pytest
import itertools
from assign_enc.matrix import *


@pytest.fixture
def gen_one_per_existence():
    src = [Node([1], repeated_allowed=False) for _ in range(2)]
    tgt = [Node([0, 1], repeated_allowed=False) for _ in range(3)]
    exist = []
    for i_src, i_tgt in itertools.product(list(range(len(src)))+[len(src)], list(range(len(tgt)))+[len(tgt)]):
        exist.append(NodeExistence(src_exists=[i == i_src for i in range(len(src))],
                                   tgt_exists=[i == i_tgt for i in range(len(tgt))]))
    return AggregateAssignmentMatrixGenerator(src=src, tgt=tgt, existence_patterns=NodeExistencePatterns(exist))
