import numpy as np
from assign_enc.encoding import *

__all__ = ['GroupedEncoder', 'GroupByIndexEncoder']


class GroupedEncoder(Encoder):
    """
    Base class for an encoder that recursively separates the matrix based on some grouping criteria, until all
    matrices are associated to one unique design vector.

    By default, grouping values are normalized within sub-groups, leading to much less value options per design
    variable. To disable, set normalize_within_group=False.
    """

    def __init__(self, *args, normalize_within_group=True, **kwargs):
        self.normalize_within_group = normalize_within_group
        super().__init__(*args, **kwargs)

    def _encode(self, matrix: np.ndarray) -> np.ndarray:
        # Do any preparations if needed
        self._prepare_grouping(matrix)

        # Determine worst-case scenario of number of design variables as a fail-safe
        n_dv_max = matrix.shape[1]*matrix.shape[2]*2

        # Add design variables until all sub-groups only consist of one matrix
        matrices = [(matrix, np.arange(0, matrix.shape[0]))]
        n_mat = matrix.shape[0]
        des_var_values = []
        dv_idx = 0
        normalize_within_group = self.normalize_within_group
        for _ in range(n_dv_max):
            # Prepare new design vector column
            des_var_values_i = np.zeros((n_mat,), dtype=int)

            # For each current sub-matrix
            next_matrices = []
            no_grouping_needed = True
            for sub_matrix, i_sub_matrix in matrices:
                # Only group sub-matrices that are non unique (i.e. more than 1)
                if sub_matrix.shape[0] <= 1:
                    next_matrices.append((sub_matrix, i_sub_matrix))

                    # Determine grouping value if we don't normalize within the group
                    if not normalize_within_group:
                        grouping_values = self._get_grouping_criteria(dv_idx, sub_matrix, i_sub_matrix)
                        des_var_values_i[i_sub_matrix] = grouping_values[0]

                else:
                    # Flag that we still have non-unique matrices
                    no_grouping_needed = False

                    # Get values to group by
                    grouping_values = self._get_grouping_criteria(dv_idx, sub_matrix, i_sub_matrix)

                    # Determine unique values
                    unique_values = np.sort(np.unique(grouping_values))
                    dv_values = np.arange(0, len(unique_values)) if normalize_within_group else unique_values
                    for i_val, unique_value in enumerate(unique_values):
                        # Map unique values to design variable values
                        i_value_mask, = np.where(grouping_values == unique_value)
                        i_value_sub_matrix = i_sub_matrix[i_value_mask]
                        des_var_values_i[i_value_sub_matrix] = dv_values[i_val]

                        # Separate next level of sub matrices
                        next_matrices.append((sub_matrix[i_value_mask, :, :], i_value_sub_matrix))

            # If all design variable values are the same, we can skip this design variable
            if len(np.unique(des_var_values_i)) > 1:
                des_var_values.append(des_var_values_i)

            # If there are no unique values and we did no grouping, we are done
            elif no_grouping_needed:
                break

            # Prepare next step
            dv_idx += 1
            matrices = next_matrices

        # The loop was finished because it ran out of design variables (not because it reached the status of all-unique)
        else:
            raise RuntimeError('Too many design variables!')

        # Concatenate design variable values into design vectors and normalize
        design_vectors = np.column_stack(des_var_values)
        if normalize_within_group:
            return design_vectors
        return self._normalize_design_vectors(design_vectors)

    def _prepare_grouping(self, matrix: np.ndarray):
        """Implement any logic needed for preparing the grouping process here"""
        pass

    def _get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        """Given a design variable index and associated sub-matrix, return a vector of values corresponding to each
        matrix to be grouped by"""
        raise NotImplementedError


class GroupByIndexEncoder(GroupedEncoder):
    """Grouping encoder that recursively separates the remaining matrices in n groups."""

    def __init__(self, *args, n_groups=2, **kwargs):
        self.n_groups = max(n_groups, 2)
        super().__init__(*args, **kwargs)

    def _get_grouping_criteria(self, dv_idx: int, sub_matrix: np.ndarray, i_sub_matrix: np.ndarray) -> np.ndarray:
        n_matrix = sub_matrix.shape[0]
        n_per_group = max(n_matrix // self.n_groups, 1)

        i_start = 0
        grouping_values = np.zeros((n_matrix,), dtype=int)
        for i in range(self.n_groups):
            grouping_values[i_start:] = i
            i_start += n_per_group

        return grouping_values
