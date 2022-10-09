import numpy as np
from assign_enc.eager.encodings.grouped_base import *

__all__ = ['ElementGroupedEncoder']


class ElementGroupedEncoder(GroupedEncoder):
    """Group by the value of each design variable"""

    def _get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        j = dv_idx // sub_matrix.shape[1]
        i = dv_idx-(j*sub_matrix.shape[1])
        return sub_matrix[:, i, j]
