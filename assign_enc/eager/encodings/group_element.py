import numpy as np
from assign_enc.eager.encodings.grouped_base import *

__all__ = ['ElementGroupedEncoder']


class ElementGroupedEncoder(GroupedEncoder):
    """Group by the value of each design variable"""

    def _get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        return self.flatten_matrix(matrix)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r})'

    def __str__(self):
        normalize_str = ' Norm Grp' if self.normalize_within_group else ''
        return f'Group By Element{normalize_str} + {self._imputer!s}'
