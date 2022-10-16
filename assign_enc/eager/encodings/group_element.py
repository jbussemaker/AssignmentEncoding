import numpy as np
from assign_enc.eager.encodings.grouped_base import *

__all__ = ['ElementGroupedEncoder']


class ElementGroupedEncoder(GroupedEncoder):
    """Group by the value of each design variable"""

    def _get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        return self.flatten_matrix(matrix)
