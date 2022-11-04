import itertools
import numpy as np
from typing import *
from assign_enc.lazy_encoding import *

__all__ = ['LazyFirstImputer']


class LazyFirstImputer(LazyImputer):
    """Imputer that chooses the first possible matrix."""

    def _impute(self, vector: DesignVector, matrix: Optional[np.ndarray], existence: NodeExistence,
                validate: Callable[[np.ndarray], bool], tried_vectors: Set[Tuple[int, ...]]) -> Tuple[DesignVector, np.ndarray]:

        # Loop through possible design vectors until one is found that has not been tried yet
        for dv in itertools.product(*[list(range(dv.n_opts)) for dv in self._des_vars]):
            dv = tuple(dv)
            if dv in tried_vectors:
                continue
            break
        else:
            raise RuntimeError('No valid design vector found!')

        # Validate this design vector and associated matrix
        vector = np.array(dv)
        matrix = self._decode(vector, existence)
        if validate(matrix):
            return vector, matrix

        # Otherwise, continue imputation
        return self.impute(vector, matrix, existence, tried_vectors)

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'First Imp'
