import numpy as np
from assign_enc.encoding import *

__all__ = ['GroupedEncoder', 'GroupByIndexEncoder']


class GroupedEncoder(Encoder):
    """Base class for an encoder that recursively separates the matrix based on some grouping criteria, until all
    matrices are associated to one unique design vector."""

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        # Do any preparations if needed
        self._prepare_grouping(matrix)

        # Determine worst-case scenario of number of design variables as a fail-safe
        n_dv_max = matrix.shape[1]*matrix.shape[2]

        # Add design variables until all sub-groups only consist of one matrix
        matrices = [matrix]
        n_mat = matrix.shape[0]
        des_var_values = []
        dv_idx = 0
        for _ in range(n_dv_max):
            # Prepare new design vector column
            des_var_values_i = np.zeros((n_mat,), dtype=int)

            # For each current sub-matrix
            next_matrices = []
            all_unique_matrices = True
            i_dv_start = 0
            for sub_matrix in matrices:
                # Only group sub-matrices that are non unique (i.e. more than 1)
                if sub_matrix.shape[0] <= 1:
                    next_matrices.append(sub_matrix)

                else:
                    # Flag that we still have non-unique matrices
                    all_unique_matrices = False

                    # Get values to group by
                    grouping_values = self._get_grouping_criteria(dv_idx, sub_matrix)

                    # Determine unique values
                    unique_values = np.sort(np.unique(grouping_values))
                    dv_values = np.arange(0, len(unique_values))
                    for i_val, unique_value in enumerate(unique_values):
                        # Map unique values to design variable values
                        i_value_mask, = np.where(grouping_values == unique_value)
                        des_var_values_i[i_value_mask+i_dv_start] = dv_values[i_val]

                        # Separate next level of sub matrices
                        next_matrices.append(sub_matrix[i_value_mask, :, :])

                # Increment design vectors starting index
                i_dv_start += sub_matrix.shape[0]

            # If all sub-matrices are unique we are done
            if all_unique_matrices:
                break

            # If all design variable values are the same, we can skip this design variable
            if len(np.unique(des_var_values_i)) > 1:
                des_var_values.append(des_var_values_i)

            # Prepare next step
            dv_idx += 1
            matrices = next_matrices

        # The loop was finished because it ran out of design variables (not because it reached the status of all-unique)
        else:
            raise RuntimeError('Too many design variables!')

        # Concatenate design variable values into design vectors
        return np.column_stack(des_var_values)

    def _prepare_grouping(self, matrix: np.ndarray):
        """Implement any logic needed for preparing the grouping process here"""
        pass

    def _get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray) -> np.ndarray:
        """Given a design variable index and associated sub-matrix, return a vector of values corresponding to each
        matrix to be grouped by"""
        raise NotImplementedError


class GroupByIndexEncoder(GroupedEncoder):
    """Grouping encoder that recursively separates the remaining matrices in n groups."""

    def __init__(self, *args, n_groups=2, **kwargs):
        self.n_groups = max(n_groups, 2)
        super().__init__(*args, **kwargs)

    def _get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray) -> np.ndarray:
        n_matrix = sub_matrix.shape[0]
        n_per_group = max(n_matrix // self.n_groups, 1)

        i_start = 0
        grouping_values = np.zeros((n_matrix,), dtype=int)
        for i in range(self.n_groups):
            grouping_values[i_start:] = i
            i_start += n_per_group

        return grouping_values
