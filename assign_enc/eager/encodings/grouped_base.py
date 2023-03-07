import itertools
import numpy as np
from typing import Optional
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
            self._get_grouping_values(matrix), normalize_within_group=self.normalize_within_group,
            ordinal_base=self._get_ordinal_conv_base())

    def _get_ordinal_conv_base(self) -> Optional[int]:
        pass

    @classmethod
    def group_by_values(cls, group_by_values: np.ndarray, normalize_within_group=True,
                        ordinal_base: int = None) -> np.ndarray:
        """
        Get design vectors that uniquely map to different value combinations. Example:
        [[1 2],      [[0 0],
         [1 3],  -->  [0 1],
         [2 2],       [1 -1],
         [3 2],       [2 -1],  # -1 means inactive

        Optionally normalize only after determining all groups.
        Optionally convert the grouped design vector values to some other base,
        e.g. for base 2: [[1, 2, 3, 4]].T --> [[0, 0], [0, 1], [1, 0], [1, 1]]"""
        design_vectors = np.empty(group_by_values.shape, dtype=int)
        row_mask_list = [np.ones((group_by_values.shape[0],), dtype=bool)]

        # Loop over columns
        for i_col in range(group_by_values.shape[1]):

            # Loop over current sub-divisions
            next_row_masks = []
            grouping_needed = False
            for row_mask in row_mask_list:

                # Check if there are multiple design vectors in this sub-division
                group_by_values_sub = group_by_values[row_mask, i_col]
                if len(group_by_values_sub) == 1:
                    next_row_masks.append(row_mask)
                    design_vectors[row_mask, i_col] = X_INACTIVE_VALUE
                    continue

                # Check if there are multiple unique values in this sub-division
                unique_values = sorted(list(set(group_by_values_sub)))
                if len(unique_values) == 1:
                    next_row_masks.append(row_mask)
                    design_vectors[row_mask, i_col] = X_INACTIVE_VALUE
                    grouping_needed = True
                    continue

                # Loop over unique values in sub-divisions
                for value_idx, value in enumerate(unique_values):
                    if value < 0:
                        raise ValueError('Values to group by should not contain negative values!')

                    # Assign indices for each unique value
                    next_row_mask = row_mask.copy()
                    next_row_mask[row_mask] = next_row_sub_mask = group_by_values_sub == value
                    design_vectors[next_row_mask, i_col] = value_idx if normalize_within_group else value
                    next_row_masks.append(next_row_mask)

                    if np.count_nonzero(next_row_sub_mask) > 1:
                        grouping_needed = True

            # Stop grouping if not needed anymore
            if not grouping_needed:
                design_vectors = design_vectors[:, :i_col+1]
                break

            row_mask_list = next_row_masks

        # Normalize design vectors
        if not normalize_within_group:
            design_vectors = cls.normalize_design_vectors(design_vectors)

        # Remove columns where there are no alternatives
        has_alternatives = np.any(design_vectors > 0, axis=0)
        design_vectors = design_vectors[:, has_alternatives]

        # Convert to other base
        if ordinal_base is not None:
            design_vectors = cls.convert_to_base(design_vectors, ordinal_base)
            has_alternatives = np.any(design_vectors > 0, axis=0)
            design_vectors = design_vectors[:, has_alternatives]

        return design_vectors

    @staticmethod
    def convert_to_base(values: np.ndarray, base: int = 2) -> np.ndarray:
        """Convert the ordinal-encoded values to another base"""
        if base < 2 or base > 9:
            raise ValueError('Base should be between 2 and 9')
        if values.shape[1] == 0:
            return values

        columns = []
        for i_col in range(values.shape[1]):
            col_values = values[:, i_col]
            active_mask = col_values != X_INACTIVE_VALUE

            unique, unique_idx_active = np.unique(col_values[active_mask], return_inverse=True)
            unique_idx = np.ones((len(col_values),))*X_INACTIVE_VALUE
            unique_idx[active_mask] = unique_idx_active

            n_converted = len(np.base_repr(len(unique)-1))
            col_converted = np.zeros((values.shape[0], n_converted), dtype=int)
            col_converted[~active_mask, :] = X_INACTIVE_VALUE

            for i_value in range(len(unique)):
                base_converted = [
                    int(char) for char in (np.base_repr(i_value, base=base) if base != 2 else np.binary_repr(i_value))]
                col_converted[unique_idx == i_value, -len(base_converted):] = base_converted

            columns.append(col_converted)
        return np.column_stack(columns)

    def _get_grouping_values(self, matrix: np.ndarray) -> np.ndarray:
        """Get a n_mat x n_dv matrix with values to sub-divide design variables by"""
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
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

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r}, n_groups={self.n_groups})'

    def __str__(self):
        return f'Group By Index ({self.n_groups} groups) + {self._imputer!s}'
