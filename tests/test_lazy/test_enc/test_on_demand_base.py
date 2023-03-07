import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_enc.lazy.encodings.on_demand_base import *
from assign_enc.lazy.imputation.constraint_violation import *


class DummyOnDemandLazyEncoder(OnDemandLazyEncoder):

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        return [DiscreteDV(n_opts=2)]

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[Tuple[DesignVector, np.ndarray]]:
        return vector, np.zeros((len(self._matrix_gen.src), len(self._matrix_gen.tgt)))


def test_on_demand_lazy_encoder():
    encoder = DummyOnDemandLazyEncoder(LazyConstraintViolationImputer())
    encoder.set_settings(MatrixGenSettings(src=[Node([0, 1, 2]), Node(min_conn=0)], tgt=[Node([0, 1]), Node(min_conn=1)]))

    n_src_n_tgt = list(encoder.iter_n_src_n_tgt())
    assert len(n_src_n_tgt) == 19

    assert n_src_n_tgt[0] == ((0, 1), (0, 1), NodeExistence())
    n_tup = n_src_n_tgt[0][:2]

    assert len(encoder._matrix_cache) == 0
    matrix = encoder.get_matrices(*n_tup)
    assert matrix.shape == (1, 2, 2)
    assert np.all(matrix == np.array([[[0, 0], [0, 1]]]))
    assert len(encoder._matrix_cache) == 1
    assert encoder.count_matrices(*n_tup) == 1
