import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.lazy_encoding import *
from assign_enc.patterns.encoder import *

__all__ = ['CombiningPatternEncoder', 'AssigningPatternEncoder', 'PartitioningPatternEncoder',
           'DownselectingPatternEncoder', 'ConnectingPatternEncoder', 'PermutingPatternEncoder',
           'UnorderedNonReplacingCombiningPatternEncoder', 'UnorderedCombiningPatternEncoder']


class CombiningPatternEncoder(PatternEncoderBase):
    """
    Encodes the combining pattern: choosing one of N options.

    Source nodes: 1 node with 1 connection
    Target nodes: N nodes with 0 or 1 connection
    Encoded as:   1 design variable with n_tgt options
    """

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        src, tgt = effective_settings.src, effective_settings.tgt

        # There may be no excluded edges
        if effective_settings.get_excluded_indices():
            return False

        # Check 1 source node with 1 connection
        if len(src) != 1 or src[0].conns != [1]:
            return False

        # Check all target nodes have 0 or 1 connections
        if any(n.conns != [0, 1] for n in tgt):
            return False

        return True

    def _encode_effective(self, effective_settings: MatrixGenSettings) -> List[DiscreteDV]:
        # One design variable with n_tgt options
        if len(effective_settings.tgt) < 2:
            return []
        return [DiscreteDV(n_opts=len(effective_settings.tgt))]

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings) \
            -> Tuple[DesignVector, np.ndarray]:
        # Determine which target is selected
        n_tgt = len(effective_settings.tgt)
        if len(vector) == 0:
            i_select = 0
        else:
            i_select = vector[0]
            if i_select >= n_tgt:
                i_select = n_tgt
                vector = [i_select]

        # Create 1 x n_tgt matrix
        matrix = np.zeros((1, n_tgt), dtype=int)
        matrix[0, i_select] = 1
        return vector, matrix

    def _pattern_name(self) -> str:
        return 'Combining'


class AssigningPatternEncoder(PatternEncoderBase):
    """
    Encodes the assigning pattern: connect n_src nodes to n_tgt nodes.
    Optionally surjective: each target node has min 1 connection.
    Optionally repeatable: connections between src and tgt nodes may be repeated.

    Source nodes: n_src nodes with any nr of connections
    Target nodes: n_tgt nodes with any nr of connections
                  (1 or more if surjective, 1 if inject and surjective)
    Encoded as:   n_src x n_tgt design variables with 2 options each (max_conn_parallel if repeatable)
    """

    def __init__(self, imputer):
        self.surjective = False  # Each tgt min 1 conn
        self.repeatable = False
        self._n_max = 1
        super().__init__(imputer)

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        if initialize:
            self.surjective = self.repeatable = False
        src, tgt = effective_settings.src, effective_settings.tgt

        # There may be no excluded edges
        if effective_settings.get_excluded_indices():
            return False

        def _set_check(attr, value):
            if initialize:
                setattr(self, attr, value)
                return True

            return getattr(self, attr) == value

        # Check if all repeatability flags are the same
        repeatable = np.array([n.rep for n in src]+[n.rep for n in tgt])
        if not np.all(repeatable) and not np.all(~repeatable):
            return False
        if not _set_check('repeatable', repeatable[0]):
            return False

        # Check all source nodes have any nr of connections
        if any(n.min_conns != 0 for n in src):
            return False

        # Check if all max parallel connections are the same
        n_max = effective_settings.get_max_conn_parallel() if self.repeatable else 1
        if not _set_check('_n_max', n_max):
            return False

        # Check if surjective (tgt nodes at least 1 conn)
        if all(n.max_inf for n in tgt):
            n_min_conn = np.array([n.min_conns for n in tgt])
            if np.all(n_min_conn == n_min_conn[0]) and n_min_conn[0] in [0, 1]:
                if not _set_check('surjective', n_min_conn[0] == 1):
                    return False
                return True

            return False

        # Do not check for bijective, as that is actually the same as the non-covering partitioning pattern,
        # so there is a better encoder available
        # Do not check for injective, as there the downselecting encoder is better

        return False

    def _encode_effective(self, effective_settings: MatrixGenSettings) -> List[DiscreteDV]:
        n_max = self._n_max
        return [DiscreteDV(n_opts=n_max+1) for _ in range(len(effective_settings.src)*len(effective_settings.tgt))]

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings) \
            -> Tuple[DesignVector, np.ndarray]:
        surjective, repeatable = self.surjective, self.repeatable
        n_src, n_tgt, n_max = len(effective_settings.src), len(effective_settings.tgt), self._n_max

        # Correct the specified number of connections
        vector = np.array(vector)
        for i_tgt in range(n_tgt):
            # Get the connections for the given target node
            conns = vector[i_tgt::n_tgt]
            n_conns = np.sum(conns)

            if surjective and n_conns < 1:
                # If surjective, there should be at least 1 connection per target: randomly set one connection
                conns[np.random.choice(np.arange(n_src))] = 1

            elif repeatable:
                # If repeatable, cap it to the max allowed parallel connections
                conns[conns > n_max] = n_max

        # The matrix is simply the reshaped design vector
        matrix = np.array(vector, dtype=int).reshape((n_src, n_tgt))
        return vector, matrix

    def _pattern_name(self) -> str:
        return 'Assigning'


class PartitioningPatternEncoder(PatternEncoderBase):
    """
    Encodes the partitioning pattern: partition M nodes into max N sets.

    Source nodes: N nodes with any nr of connections
    Target nodes: M nodes with 1 connection
    Encoded as:   M design variables with N options each
    """

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        src, tgt = effective_settings.src, effective_settings.tgt

        # There may be no excluded edges
        if effective_settings.get_excluded_indices():
            return False

        # Check if all source nodes have any nr of connections
        if any(n.min_conns != 0 for n in src):
            return False

        # Check if all target nodes have 1 connection
        if any(n.conns != [1] for n in tgt):
            return False

        return True

    def _encode_effective(self, effective_settings: MatrixGenSettings) -> List[DiscreteDV]:
        return [DiscreteDV(n_opts=len(effective_settings.src)) for _ in range(len(effective_settings.tgt))]

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings) \
            -> Tuple[DesignVector, np.ndarray]:

        # Set connections to source nodes
        n_src, n_tgt = len(effective_settings.src), len(effective_settings.tgt)
        matrix = np.zeros((n_src, n_tgt), dtype=int)
        for i in range(n_tgt):
            matrix[vector[i], i] = 1

        return vector, matrix

    def _pattern_name(self) -> str:
        return 'Partitioning'


class DownselectingPatternEncoder(PatternEncoderBase):
    """
    Encodes the downselecting pattern: select any from N nodes
    Encodes the injective assigning pattern: connect M src nodes to N target nodes that accept 0 or 1 connections

    Source nodes: M nodes (1 if downselecting) with any nr of connections
    Target nodes: N nodes with 0 or 1 connections
    Encoded as:   N design variables with M+1 options (i.e. binary if downselecting)
    """

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        src, tgt = effective_settings.src, effective_settings.tgt

        # There may be no excluded edges
        if effective_settings.get_excluded_indices():
            return False

        # Check if there is all source nodes have any nr of connections
        if any(n.min_conns != 0 for n in src):
            return False

        # Check if all target nodes have 0 or 1 connections
        if any(n.conns != [0, 1] for n in tgt):
            return False

        return True

    def _encode_effective(self, effective_settings: MatrixGenSettings) -> List[DiscreteDV]:
        n_opts = len(effective_settings.src)+1
        return [DiscreteDV(n_opts=n_opts) for _ in range(len(effective_settings.tgt))]

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings) \
            -> Tuple[DesignVector, np.ndarray]:

        # Each variable in the vector corresponds to a target, where 0 means no connection and >0 means a connection to
        # the ith src
        matrix = np.zeros((len(effective_settings.src), len(effective_settings.tgt)), dtype=int)
        for i_tgt in range(matrix.shape[1]):
            i_src = vector[i_tgt]
            if i_src > 0:
                matrix[i_src-1, i_tgt] = 1
        return vector, matrix

    def _pattern_name(self) -> str:
        return 'Downselecting'


class ConnectingPatternEncoder(PatternEncoderBase):
    """
    Encodes the connecting pattern: connect N src to N tgt nodes, however not to the same indices
    Optionally undirected: only connect one triangle of the matrix (as each connection is seen as bi-directional)

    Source nodes: N nodes with any nr of connections
    Target nodes: N nodes with any nr of connections
    Excluded:     (i, j) if i == j (if directed); (i, j) if i >= j (if undirected)
    Encoded as:   N*(N-1) binary design variables if directed; (N*(N-1))/2 if undirected
    """

    def __init__(self, imputer):
        self.directed = False
        super().__init__(imputer)

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        src, tgt = effective_settings.src, effective_settings.tgt

        # Check if there are the same nr of src and tgt nodes
        if len(src) == 1 or len(src) != len(tgt):
            return False

        # Check if all nodes have any nr of connections
        if any(n.min_conns != 0 for n in src) or any(n.min_conns != 0 for n in tgt):
            return False

        # Check if all nodes are non-repeatable
        if any(n.rep for n in src) or any(n.rep for n in tgt):
            return False

        # Check if there are any excluded edges
        excluded = set(effective_settings.get_excluded_indices())
        if not excluded:
            return False

        # Check if the diagonal is excluded
        if any((i, i) not in excluded for i in range(len(src))):
            return False

        # Check if nothing from the upper triangle is excluded
        if any((i, j) in excluded for i in range(len(src)) for j in range(len(tgt)) if i < j):
            return False

        # Check if also the lower triangle is excluded
        lower_tri_excluded = np.array([(i, j) in excluded for i in range(len(src)) for j in range(len(tgt)) if i > j])
        if np.all(lower_tri_excluded):
            if initialize:
                self.directed = False
            return not self.directed

        # Check if the lower triangle is included
        if np.all(~lower_tri_excluded):
            if initialize:
                self.directed = True
            return self.directed

        return False

    def _encode_effective(self, effective_settings: MatrixGenSettings) -> List[DiscreteDV]:
        # Number of design variables if n*n minus n diagonal connections --> n*(n-1)
        n_src = len(effective_settings.src)
        n_dv = n_src*(n_src-1)

        # If not directed, also the lower triangle cannot be connected two, so we half the nr of design variables
        if not self.directed:
            n_dv = int(n_dv/2)

        return [DiscreteDV(n_opts=2) for _ in range(n_dv)]

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings) \
            -> Tuple[DesignVector, np.ndarray]:

        # Fill the upper triangle of the matrix
        n_src = len(effective_settings.src)
        matrix = np.zeros((n_src, n_src), dtype=int)
        n_upper = int(len(vector)/2) if self.directed else len(vector)
        matrix[np.triu_indices(n_src, k=1)] = vector[:n_upper]

        # If directed, also fill the lower triangle
        if self.directed:
            matrix[np.tril_indices(n_src, k=-1)] = vector[n_upper:]

        return vector, matrix

    def _pattern_name(self) -> str:
        return 'Connecting'


class PermutingPatternEncoder(PatternEncoderBase):
    """
    Encodes the permuting pattern: shuffle the order of N elements

    Source nodes: N nodes with 1 connection
    Target nodes: N nodes with 1 connection
    Encoded as:   N-1 design variables with N-i options
    """

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        src, tgt = effective_settings.src, effective_settings.tgt

        # There may be no excluded edges
        if effective_settings.get_excluded_indices():
            return False

        # Check if there are the same nr of src and tgt nodes
        if len(src) == 1 or len(src) != len(tgt):
            return False

        # Check if all source nodes have 1 connection
        if any(n.conns != [1] for n in src):
            return False

        # Check if all target nodes have 1 connection
        if any(n.conns != [1] for n in tgt):
            return False

        return True

    def _encode_effective(self, effective_settings: MatrixGenSettings) -> List[DiscreteDV]:
        n = len(effective_settings.src)
        return [DiscreteDV(n_opts=n-i) for i in range(n-1)]

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings) -> Tuple[DesignVector, np.ndarray]:
        n = len(effective_settings.src)
        matrix = np.zeros((n, n), dtype=int)

        # Set positions: the position of an element depends on the previously selected elements
        available = list(range(n))
        for i in range(n-1):
            relative_pos = vector[i]
            pos = available[relative_pos]

            matrix[i, pos] = 1
            available.remove(pos)

        # Set remaining connection
        matrix[-1, available[0]] = 1

        return vector, matrix

    def _pattern_name(self) -> str:
        return 'Permuting'


class UnorderedNonReplacingCombiningPatternEncoder(PatternEncoderBase):
    """
    Encodes the unordered combining pattern: take M from N elements, no replacement

    Source nodes: 1 node with M connections
    Target nodes: N nodes with 0 or 1 connection
    Encoded as:   N-1 binary design variables
    """

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        src, tgt = effective_settings.src, effective_settings.tgt

        # There may be no excluded edges
        if effective_settings.get_excluded_indices():
            return False

        # Check if there is one node with 1 specific amount of connections (other than 1)
        if len(src) != 1 or not src[0].conns or len(src[0].conns) != 1 or src[0].conns == [1]:
            return False

        # Check if all target nodes have 0 or 1 connection
        if all(n.conns == [0, 1] for n in tgt):
            return True

        return False

    def _encode_effective(self, effective_settings: MatrixGenSettings) -> List[DiscreteDV]:
        n_tgt = len(effective_settings.tgt)
        return [DiscreteDV(n_opts=2) for _ in range(n_tgt-1)]

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings) \
            -> Tuple[DesignVector, np.ndarray]:
        n_take = effective_settings.src[0].conns[0]
        n_tgt = len(effective_settings.tgt)

        # Ensure we have selected the correct number of elements
        # Note that we have one design variable less than the number of elements that can be selected: the last element
        # is selected if there n_take-1 elements are taken
        vector = np.array(vector)
        n_taken = np.sum(vector)
        if n_taken < n_take-1:
            # Randomly select elements
            i_not_selected = np.where(vector == 0)[0]
            target = (n_take-1) if np.random.random() > .5 else n_take
            vector[np.random.choice(i_not_selected, target-n_taken, replace=False)] = 1

        elif n_taken > n_take:
            # Randomly deselect elements
            i_selected = np.where(vector == 1)[0]
            target = (n_take-1) if np.random.random() > .5 else n_take
            vector[np.random.choice(i_selected, n_taken-target, replace=False)] = 0

        # Set matrix connections in the first row
        matrix = np.zeros((1, n_tgt), dtype=int)
        matrix[0, :len(vector)] = vector

        # Select last element if nr of taken elements is n_take-1
        if np.sum(vector) == n_take-1:
            matrix[0, -1] = 1

        return vector, matrix

    def _pattern_name(self) -> str:
        return 'Unordered Non-replacing Combining'


class UnorderedCombiningPatternEncoder(PatternEncoderBase):
    """
    Encodes the unordered combining pattern: take M from N elements, with replacement

    Source nodes: 1 node with M connections, repeatable
    Target nodes: N nodes with any nr of connections, repeatable
    Encoded as:   M design variables with N options each
    """

    def _matches_pattern(self, effective_settings: MatrixGenSettings, initialize: bool) -> bool:
        src, tgt = effective_settings.src, effective_settings.tgt

        # There may be no excluded edges
        if effective_settings.get_excluded_indices():
            return False

        # Check if there is one node with 1 specific amount of connections (other than 1)
        if len(src) != 1 or not src[0].conns or len(src[0].conns) != 1 or src[0].conns == [1]:
            return False

        # Check if all target nodes have any nr of connections
        if all(n.min_conns == 0 for n in tgt):

            # Check if all connections are repeated
            if all(n.rep for nodes in (src, tgt) for n in nodes):
                return True

        return False

    def _encode_effective(self, effective_settings: MatrixGenSettings) -> List[DiscreteDV]:
        n_take = effective_settings.src[0].conns[0]
        n_tgt = len(effective_settings.tgt)
        return [DiscreteDV(n_opts=n_tgt) for _ in range(n_take)]

    def _decode_effective(self, vector: DesignVector, effective_settings: MatrixGenSettings) \
            -> Tuple[DesignVector, np.ndarray]:
        n_take = effective_settings.src[0].conns[0]
        n_tgt = len(effective_settings.tgt)

        """
        Correct the vector by ensuring that each subsequent value is equal or higher
        This also works if we don't have replacement, as there the design vectors are encoded with an offset anyway
        
        We correct the vector starting in the center in order to be closer to the true distribution of vectors if design
        variables are generated randomly.
        For example, take 2 from 3, with replacement:
        000 001 002 011 012 022 111 112 122 222
        The amounts each value at each position occurs:
         DV: 0 1 2
        0    6 3 1
        1    3 4 3
        2    1 3 6
        As you can see, for the center design variable, the distribution of selected values is closest to uniform.
        Therefore when all design variables are randomly selected, it makes sense to start there for correcting the
        other design variables. This result can be reproduced for every take x from y.
        """

        vector = np.array(vector)
        n_max = n_tgt

        # Correct variables before the center variable
        i_center = int(np.floor(.5*(n_take-1)))
        for i in reversed(range(0, i_center)):
            if vector[i] > vector[i+1]:
                # Move to any of the positions that are equal or lower
                vector[i] = np.random.choice(np.arange(0, vector[i+1]+1))

        # Correct variables after the center variable
        for i in range(i_center+1, n_take):
            if vector[i] < vector[i-1]:
                # Move to any of the random positions that are equal or higher
                vector[i] = np.random.choice(np.arange(vector[i-1], n_max))

        # Set matrix connections in the first row
        matrix = np.zeros((1, n_tgt), dtype=int)
        for i_dv, i in enumerate(vector):
            matrix[0, i] += 1
        return vector, matrix

    def _pattern_name(self) -> str:
        return 'Unordered Combining'
