import string
import numpy as np
from typing import *
from assign_enc.lazy_encoding import *

__all__ = ['EnumRecursiveEncoder']


class EnumRecursiveEncoder(QuasiLazyEncoder):
    """Defines design variables where each has a fixed nr of options to cover all possible assignment patterns."""

    _str_lookup = {char: i for i, char in enumerate(string.digits+string.ascii_uppercase)}

    def __init__(self, *args, n_divide=2, **kwargs):
        self.n_divide = max(2, n_divide)
        super().__init__(*args, **kwargs)
        self._dv_last = None
        self._dv_inactive_key = None

    def _encode_prepare(self):
        super()._encode_prepare()
        self._dv_last = None
        self._dv_inactive_key = None

    def _encode_matrix(self, matrix: np.ndarray) -> List[DiscreteDV]:
        n_mat = matrix.shape[0]
        if n_mat <= 1:
            return []

        n = self.n_divide
        if n_mat < n:
            return [DiscreteDV(n_opts=n_mat)]

        n_var = int(np.ceil(np.log(n_mat)/np.log(n)))

        # Get design vector values that lead to inactive variables (due to nr cutoff)
        dv_last = np.array(self.base_repr_int(n_mat-1, n))
        i_inactive = np.where(dv_last == 0)[0]
        if len(i_inactive) > 0:
            left_side_values = dv_last[:i_inactive[-1]+1]
            self._dv_inactive_key = (i_inactive, left_side_values)
            dv_last[i_inactive] = X_INACTIVE_VALUE
        self._dv_last = dv_last

        return [DiscreteDV(n_opts=n) for _ in range(n_var)]

    def _decode_matrix(self, vector: DesignVector, matrix: np.ndarray) -> Optional[Tuple[DesignVector, np.ndarray]]:
        if len(vector) == 0:
            return vector, matrix[0, :, :]

        i_mat = np.sum((self.n_divide**np.arange(len(vector)))*vector[::-1])
        if i_mat >= matrix.shape[0]:
            return self._dv_last, matrix[-1, :, :]

        if self._dv_inactive_key is not None:
            i_inactive, left_side_values = self._dv_inactive_key
            if np.all(vector[:len(left_side_values)] == left_side_values):
                vector = np.array(vector)
                vector[i_inactive] = X_INACTIVE_VALUE

        return vector, matrix[i_mat, :, :]

    @classmethod
    def base_repr_int(cls, value: int, base: int) -> List[int]:
        return [cls._str_lookup[char] for char in (np.binary_repr(value) if base == 2 else np.base_repr(value, base))]

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r})'

    def __str__(self):
        return f'Enum Rec {self.n_divide} + {self._imputer!s}'
