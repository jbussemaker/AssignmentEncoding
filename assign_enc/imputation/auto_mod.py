import numpy as np
from typing import *
from assign_enc.encoding import *

__all__ = ['AutoModImputer']


class AutoModImputer(Imputer):

    def __init__(self, reverse=False):
        super().__init__()
        self.reverse = reverse

    def impute(self, vector: DesignVector, matrix_mask: MatrixSelectMask) -> Tuple[DesignVector, np.ndarray]:
        mask = matrix_mask
        partial_vector: PartialDesignVector = [None for _ in range(len(vector))]
        for i_dv in (reversed(range(len(vector))) if self.reverse else range(len(vector))):
            dv_partial_vector = partial_vector.copy()
            dv_partial_vector[i_dv] = vector[i_dv]

            # Find design vectors that match the new partial design vector
            dv_mask = self._filter_design_vectors(dv_partial_vector) & mask
            i_mask, = np.where(dv_mask)

            # If we have found no valid design vectors for this partial vector, modify the value until we have found one
            if len(i_mask) == 0:
                for dv_value in range(self._design_vars[i_dv].n_opts):
                    # Skip current value
                    if dv_value == vector[i_dv]:
                        continue

                    # Find design vectors that match the modified partial design vector
                    dv_partial_vector[i_dv] = dv_value
                    dv_mask = self._filter_design_vectors(dv_partial_vector) & mask
                    i_mask, = np.where(dv_mask)

                    if len(i_mask) > 0:
                        break

                if len(i_mask) == 0:
                    raise RuntimeError('Cannot find any valid design vectors!')

            # If we have found exactly one vector, we can stop searching and return this one
            if len(i_mask) == 1:
                return self._return_imputation(i_mask[0])

            # Otherwise, continue with the next design variable
            else:
                mask = dv_mask
                partial_vector = dv_partial_vector

        # Only happens if there are duplicate design vectors
        raise RuntimeError('Multiple possible design vectors found! Check if there are any duplicates')
