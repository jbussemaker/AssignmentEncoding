import itertools
import numpy as np
from assign_enc.encoding import *

__all__ = ['GroupedEncoder', 'GroupByIndexEncoder']


class GroupedEncoder(EagerEncoder):
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
        return self.group_by_values(
            self._get_grouping_values(matrix), normalize_within_group=self.normalize_within_group)

    @classmethod
    def group_by_values(cls, group_by_values: np.ndarray, normalize_within_group=True) -> np.ndarray:
        """
        Get design vectors that uniquely map to different value combinations. Example:
        [[1 2],      [[0 0],
         [1 3],  -->  [0 1],
         [2 2],       [1 0],
         [3 2],       [2 0],

        Optionally normalize only after determining all groups.
        """
        design_vectors = np.empty(group_by_values.shape, dtype=int)
        row_mask_list = [np.ones((group_by_values.shape[0],), dtype=bool)]

        # Loop over columns
        for i_col in range(group_by_values.shape[1]):

            # Loop over current sub-divisions
            next_row_masks = []
            grouping_needed = False
            for row_mask in row_mask_list:

                # Loop over unique values in sub-divisions
                unique_values = np.sort(np.unique(group_by_values[row_mask, i_col]))
                for value_idx, value in enumerate(unique_values):

                    # Assign indices for each unique value
                    next_row_mask = row_mask & (group_by_values[:, i_col] == value)
                    design_vectors[next_row_mask, i_col] = value_idx if normalize_within_group else value
                    next_row_masks.append(next_row_mask)

                    if len(np.where(next_row_mask)[0]) > 1:
                        grouping_needed = True

            # Stop grouping if not needed anymore
            if not grouping_needed:
                design_vectors = design_vectors[:, :i_col+1]
                break

            row_mask_list = next_row_masks

        # Normalize design vectors
        if not normalize_within_group:
            design_vectors = cls._normalize_design_vectors(design_vectors)

        # Remove columns where there are no alternatives
        has_alternatives = np.any(design_vectors > 0, axis=0)
        design_vectors = design_vectors[:, has_alternatives]

        return design_vectors

    def _get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        """Get a n_mat x n_dv matrix with values to sub-divide design variables by"""
        raise NotImplementedError


class GroupByIndexEncoder(GroupedEncoder):
    """Grouping encoder that recursively separates the remaining matrices in n groups."""

    def __init__(self, *args, n_groups=2, **kwargs):
        self.n_groups = max(n_groups, 2)
        super().__init__(*args, **kwargs)

    def _get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        n_groups = self.n_groups
        n_var = int(np.ceil(np.log2(matrix.shape[0])))
        design_vectors = np.array(list(itertools.product(*[range(n_groups) for _ in range(n_var)]))[:matrix.shape[0]])
        for i_var in reversed(range(1, n_var)):
            design_vectors[:, i_var:] += np.array([n_groups*design_vectors[:, i_var-1]]).T
        return design_vectors
