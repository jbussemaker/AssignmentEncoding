import numpy as np
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_enc.lazy.encodings.on_demand_base import *
from assign_enc.lazy.imputation.constraint_violation import *


class DummyOnDemandLazyEncoder(OnDemandLazyEncoder):

    def _encode(self):
        return [DiscreteDV(n_opts=2)]

    def _decode(self, vector: DesignVector, src_exists: np.ndarray, tgt_exists: np.ndarray):
        return np.zeros((len(src_exists), len(tgt_exists)))


def test_on_demand_lazy_encoder():
    encoder = DummyOnDemandLazyEncoder(LazyConstraintViolationImputer())
    encoder.set_nodes(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)])

    n_src_n_tgt = list(encoder.iter_n_src_n_tgt())
    assert len(n_src_n_tgt) == 16

    assert n_src_n_tgt[0] == ((0, 1), (0, 1))

    assert len(encoder._matrix_cache) == 0
    matrix = encoder.get_matrices(*n_src_n_tgt[0])
    assert matrix.shape == (1, 2, 2)
    assert np.all(matrix == np.array([[[0, 0], [0, 1]]]))
    assert len(encoder._matrix_cache) == 1
    assert encoder.count_matrices(*n_src_n_tgt[0]) == 1

    assert encoder.get_matrices(*n_src_n_tgt[0], src_exists=np.array([True, False]),
                                tgt_exists=np.array([True, True])).shape[0] == 0
