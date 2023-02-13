import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_pymoo.problem import *
from pymoo.core.repair import Repair
from assign_enc.lazy_encoding import *
from pymoo.algorithms.moo.nsga2 import NSGA2

__all__ = ['AnalyticalProblemBase', 'AnalyticalCombinationProblem', 'AnalyticalAssignmentProblem',
           'AnalyticalPartitioningProblem', 'AnalyticalDownselectingProblem', 'AnalyticalConnectingProblem',
           'AnalyticalPermutingProblem', 'AnalyticalIterCombinationsProblem',
           'AnalyticalIterCombinationsReplacementProblem', 'ManualBestEncoder']


class ManualBestEncoder(LazyEncoder):
    """Manual encoder for an analytical optimization problem"""

    def _impute(self, vector, matrix, existence: NodeExistence) -> Tuple[DesignVector, np.ndarray]:
        raise RuntimeError('Manual best encoder should never (automatically) impute!')

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        """Encode the assignment problem (given by src and tgt nodes) directly to design variables"""
        raise NotImplementedError

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[np.ndarray]:
        """Return the connection matrix as would be encoded by the given design vector"""
        raise NotImplementedError

    def _pattern_name(self) -> str:
        raise NotImplementedError

    def get_nsga2(self, pop_size=100) -> Optional[NSGA2]:
        """Return optimally-configured NSGA2 to solve this encoded problem"""

    def __repr__(self):
        return f'{self.__class__.__name__}({self._imputer!r})'

    def __str__(self):
        return f'Manual Best {self._pattern_name()} Encoding'


class AnalyticalProblemBase(AssignmentProblem):
    """Test problem where the objective is calculated from the sum of the products of the coefficient of each
    connection, defined by an exponential function."""

    def __init__(self, encoder, n_src: int = 2, n_tgt: int = 3):
        self._n_src = n_src
        self._n_tgt = n_tgt
        self.coeff = self.get_coefficients(self._n_src, self._n_tgt)
        self._invert_f2 = False
        self._skew_f2 = False
        super().__init__(encoder)

    def get_init_kwargs(self) -> dict:
        return {'n_src': self._n_src, 'n_tgt': self._n_tgt}

    def get_src_tgt_nodes(self) -> Tuple[List[Node], List[Node]]:
        src_nodes = [self._get_node(True, i) for i in range(self.coeff.shape[0])]
        tgt_nodes = [self._get_node(False, i) for i in range(self.coeff.shape[1])]
        return src_nodes, tgt_nodes

    def get_n_obj(self) -> int:
        return 2

    def get_n_valid_design_points(self, n_cont=5) -> int:
        return self.assignment_manager.matrix_gen.count_all_matrices(max_by_existence=False)

    def _do_evaluate(self, conns: List[Tuple[int, int]], x_aux: Optional[DesignVector]) -> Tuple[List[float], List[float]]:
        f = np.zeros((2,))
        for i_src, i_tgt in conns:
            f += self.coeff[i_src, i_tgt, :]
        if self._invert_f2:
            f[1] = -f[1]
        if self._skew_f2:
            f[1] -= .25*f[0]
            f[0] -= .25*f[1]
        if self.n_obj == 1:
            return list(f)[:1], []
        return list(f), []

    @staticmethod
    def get_coefficients(n_src, n_tgt):
        coeff = np.ones((n_src, n_tgt, 2))
        for i in range(n_src):
            coeff[i, :, 0] = np.sin(np.linspace(0, 3*np.pi, n_tgt) + .25*np.pi*i)-1
            coeff[i, :, 1] = np.cos(np.linspace(-np.pi, 1.5*np.pi, n_tgt) - .5*np.pi*i)+1
        return coeff

    def get_repair(self):
        encoder = self.assignment_manager.encoder
        if isinstance(encoder, ManualBestEncoder):
            nsga2 = encoder.get_nsga2()
            if nsga2 is not None and nsga2.repair is not None:
                return nsga2.repair
        return super().get_repair()

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[ManualBestEncoder]:
        pass

    def _get_node(self, src: bool, idx: int) -> Node:
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError

    def get_problem_name(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class AnalyticalCombinationProblem(AnalyticalProblemBase):
    """Combination pattern: 1 source selecting one of the targets:
    source has 1 connection, targets 0 or 1, no repetitions"""

    def __init__(self, encoder, n_tgt: int = 3):
        super().__init__(encoder, n_src=1, n_tgt=n_tgt)

    def get_init_kwargs(self) -> dict:
        return {'n_tgt': self._n_tgt}

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node([1], repeated_allowed=False) if src else Node([0, 1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_tgt={self._n_tgt})'

    def get_problem_name(self):
        return 'An Comb Prob'

    def __str__(self):
        return f'{self.get_problem_name()} {self._n_src} -> {self._n_tgt}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[ManualBestEncoder]:
        return CombinationBestEncoder(imputer)


class CombinationBestEncoder(ManualBestEncoder):
    """Manually encodes the combination pattern: 1 design variable with n_tgt options [Selva2016]"""

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        return [DiscreteDV(n_opts=self.n_tgt)]

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[np.ndarray]:
        matrix = np.zeros((1, self.n_tgt), dtype=int)
        matrix[0, vector[0]] = 1
        return matrix

    def _pattern_name(self) -> str:
        return 'Combination'


class AnalyticalAssignmentProblem(AnalyticalProblemBase):
    """Assignment pattern: connect any source to any target, optionally with injective/surjective/bijective constraints:
    sources and target have any number of connections, no repetitions;
    - injective: targets have max 1 connection
    - surjective: targets have min 1 connection (note: this is the same as a covering partitioning problem!)
    - bijective (= injective & surjective): targets have exactly 1 connection (same as non-covering partitioning)"""

    def __init__(self, *args, injective=False, surjective=False, repeatable=False, **kwargs):
        self.injective = injective
        self.surjective = surjective
        self.repeatable = repeatable
        super().__init__(*args, **kwargs)
        if (injective and surjective) or repeatable:
            self._invert_f2 = True
        if repeatable:
            self._skew_f2 = True

    def get_init_kwargs(self) -> dict:
        return {**super().get_init_kwargs(), **{
            'injective': self.injective, 'surjective': self.surjective, 'repeatable': self.repeatable}}

    def _get_node(self, src: bool, idx: int) -> Node:
        if src:
            return Node(min_conn=0, repeated_allowed=self.repeatable)
        if self.injective:
            if self.surjective:
                return Node([1], repeated_allowed=self.repeatable)
            return Node([0, 1], repeated_allowed=self.repeatable)
        return Node(min_conn=1 if self.surjective else 0, repeated_allowed=self.repeatable)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_src={self._n_src}, n_tgt={self._n_tgt}, ' \
               f'injective={self.injective}, surjective={self.surjective}, repeatable={self.repeatable})'

    def get_problem_name(self):
        return f'An Assign Prob{"; inj" if self.injective else ""}{"; sur" if self.surjective else ""}' \
               f'{"; rep" if self.repeatable else ""}'

    def __str__(self):
        return f'An Assign Prob {self._n_src} -> {self._n_tgt}{"; inj" if self.injective else ""}' \
               f'{"; sur" if self.surjective else ""}{"; rep" if self.repeatable else ""}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[ManualBestEncoder]:
        if self.injective and self.surjective and not self.repeatable:
            return PartitioningBestEncoder(imputer)
        return AssigningBestEncoder(imputer, injective=self.injective, surjective=self.surjective, repeatable=self.repeatable)


class AssigningBestEncoder(ManualBestEncoder):
    """Manually encodes the assigning pattern:
    n_src x n_tgt binary design variables, if repeatable n_opts=max(n_src,n_tgt) [Selva2016]"""

    def __init__(self, imputer, injective=False, surjective=False, repeatable=False):
        self.injective = injective  # Each tgt max 1 conn
        self.surjective = surjective  # Each tgt min 1 conn
        self.repeatable = repeatable
        super().__init__(imputer)

    @property
    def _n_max(self):
        n_max = 1
        if self.repeatable:
            n_max_src = sum([max(1, node.min_conns) if node.max_inf else max(node.conns) for node in self.src])
            n_max_tgt = sum([max(1, node.min_conns) if node.max_inf else max(node.conns) for node in self.tgt])
            n_max = max(n_max_src, n_max_tgt)
        return n_max

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        n_max = self._n_max
        return [DiscreteDV(n_opts=n_max+1) for _ in range(self.n_src*self.n_tgt)]

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[np.ndarray]:
        return np.array(vector, dtype=int).reshape((self.n_src, self.n_tgt))

    def _pattern_name(self) -> str:
        return 'Assigning'

    def get_nsga2(self, pop_size=100) -> Optional[NSGA2]:
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        return NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            repair=AssigningRepair(
                self.n_src, self.n_tgt, self._n_max, injective=self.injective, surjective=self.surjective,
                repeatable=self.repeatable),
        )


class AssigningRepair(Repair):

    def __init__(self, n_src, n_tgt, n_max, injective=False, surjective=False, repeatable=False):
        self.n_src = n_src
        self.n_tgt = n_tgt
        self.n_max = n_max
        self.injective = injective
        self.surjective = surjective
        self.repeatable = repeatable
        super().__init__()

    def _do(self, problem, X, **kwargs):
        X = np.around(X).astype(int)
        injective, surjective, repeatable = self.injective, self.surjective, self.repeatable

        n_src, n_tgt, n_max = self.n_src, self.n_tgt, self.n_max
        for row in X:
            for i_tgt in range(n_tgt):
                conns = row[i_tgt::n_tgt]
                n_conns = np.sum(conns)
                if injective and n_conns > 1:
                    has_conn = conns == 1
                    conns[:] = 0
                    conns[np.random.choice(np.where(has_conn)[0])] = 1
                elif surjective and n_conns < 1:
                    conns[np.random.choice(np.arange(n_src))] = 1

                elif repeatable and n_conns > n_max:
                    while np.sum(conns) > n_max:
                        conns[np.random.choice(np.where(conns > 0)[0])] -= 1

            if repeatable:
                for i_src in range(n_src):
                    conns = row[i_src*n_tgt:(i_src+1)*n_tgt]
                    while np.sum(conns) > n_max:
                        conns[np.random.choice(np.where(conns > 0)[0])] -= 1

        return X


class AnalyticalPartitioningProblem(AnalyticalProblemBase):
    """Partitioning pattern: sources have any connections, targets 1, no repetitions; if changed into a covering
    partitioning pattern, targets have no max connections
    Note: in reverse (and non-covering), this represents multiple combination choices"""

    def __init__(self, *args, covering=False, **kwargs):
        self.covering = covering
        super().__init__(*args, **kwargs)
        if not covering:
            self._invert_f2 = True

    def get_init_kwargs(self) -> dict:
        return {**super().get_init_kwargs(), **{'covering': self.covering}}

    def _get_node(self, src: bool, idx: int) -> Node:
        if src:
            return Node(min_conn=0, repeated_allowed=False)
        return Node(min_conn=1, repeated_allowed=False) if self.covering else Node([1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_src={self._n_src}, n_tgt={self._n_tgt}, ' \
               f'covering={self.covering})'

    def get_problem_name(self):
        return f'An Part Prob{"; cov" if self.covering else ""}'

    def __str__(self):
        return f'An Part Prob {self._n_src} -> {self._n_tgt}{"; cov" if self.covering else ""}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[ManualBestEncoder]:
        if self.covering:
            return CoveringPartitioningBestEncoder(imputer)
        return PartitioningBestEncoder(imputer)


class PartitioningBestEncoder(ManualBestEncoder):
    """Manually encodes the partitioning pattern: n_tgt design variables with n_src options [Selva2016]"""

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        return [DiscreteDV(n_opts=self.n_src) for _ in range(self.n_tgt)]

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[np.ndarray]:
        vector = np.array(vector)
        matrix = np.zeros((self.n_src, self.n_tgt), dtype=int)
        for i in range(self.n_src):
            matrix[i, vector == i] = 1
        return matrix

    def _pattern_name(self) -> str:
        return 'Partitioning'


class CoveringPartitioningBestEncoder(AssigningBestEncoder):
    """Manually encodes the covering partitioning pattern: n_src*n_tgt design variables (same as assigning encoder)"""

    def __init__(self, imputer):
        super().__init__(imputer, surjective=True)


class AnalyticalDownselectingProblem(AnalyticalProblemBase):
    """Downselecting pattern: 1 source selecting one or more of the targets:
    source has any number connection, targets 0 or 1, no repetitions"""

    def __init__(self, encoder, n_tgt: int = 3):
        super().__init__(encoder, n_src=1, n_tgt=n_tgt)

    def get_init_kwargs(self) -> dict:
        return {'n_tgt': self._n_tgt}

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node(min_conn=0, repeated_allowed=False) if src else Node([0, 1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_tgt={self._n_tgt})'

    def get_problem_name(self):
        return 'An Down Prob'

    def __str__(self):
        return f'An Down Prob {self._n_src} -> {self._n_tgt}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[ManualBestEncoder]:
        return DownselectingBestEncoder(imputer)


class DownselectingBestEncoder(ManualBestEncoder):
    """Manually encodes the downselecting pattern: n_tgt binary design variables [Selva2016]"""

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        return [DiscreteDV(n_opts=2) for _ in range(self.n_tgt)]

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[np.ndarray]:
        return np.array(vector, dtype=int).reshape((1, self.n_tgt))

    def _pattern_name(self) -> str:
        return 'Downselecting'


class AnalyticalConnectingProblem(AnalyticalProblemBase):
    """Connecting pattern: connecting multiple nodes to each other (sources are the same as targets:
    sources and targets have any nr of connections, but cannot be connected to themselves, no repetitions;
    optionally undirected --> only one triangle of the matrix can be connected to"""

    def __init__(self, encoder, n: int = 3, directed=True):
        self._src_nodes = [Node(min_conn=0, repeated_allowed=False) for _ in range(n)]
        self._tgt_nodes = [Node(min_conn=0, repeated_allowed=False) for _ in range(n)]
        self._n = n
        self.directed = directed
        super().__init__(encoder, n_src=n, n_tgt=n)

    def get_init_kwargs(self) -> dict:
        return {'n': self._n, 'directed': self.directed}

    def _get_node(self, src: bool, idx: int) -> Node:
        return self._src_nodes[idx] if src else self._tgt_nodes[idx]

    def get_excluded_edges(self) -> Optional[List[Tuple[Node, Node]]]:
        no_diagonal = [(self._src_nodes[i], self._tgt_nodes[i]) for i in range(self._n)]
        if self.directed:
            return no_diagonal
        no_lower_triangle = [(self._src_nodes[i], self._tgt_nodes[j])
                             for i in range(self._n) for j in range(self._n) if i > j]
        return no_diagonal+no_lower_triangle

    def __repr__(self):
        return f'{self.__class__.__name__}(n={self._n_src}, directed={self.directed})'

    def get_problem_name(self):
        return f'An Conn Prob{"; dir" if self.directed else ""}'

    def __str__(self):
        return f'An Conn Prob {self._n_src} -> {self._n_tgt}{"; dir" if self.directed else ""}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[ManualBestEncoder]:
        return ConnectingBestEncoder(imputer, directed=self.directed)


class ConnectingBestEncoder(ManualBestEncoder):
    """Manually encodes the connecting pattern:
    n*(n-1) binary design variables if directed, else (n*(n-1))/2 binary design variables [Selva2016]"""

    def __init__(self, imputer, directed=True):
        self.directed = directed
        super().__init__(imputer)

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        n_dv = self.n_src*(self.n_src-1)
        if not self.directed:
            n_dv = int(n_dv/2)
        return [DiscreteDV(n_opts=2) for _ in range(n_dv)]

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[np.ndarray]:
        matrix = np.zeros((self.n_src, self.n_src), dtype=int)
        n_upper = int(len(vector)/2) if self.directed else len(vector)
        matrix[np.triu_indices(self.n_src, k=1)] = vector[:n_upper]
        if self.directed:
            matrix[np.tril_indices(self.n_src, k=-1)] = vector[n_upper:]
        return matrix

    def _pattern_name(self) -> str:
        return 'Connecting'


class AnalyticalPermutingProblem(AnalyticalProblemBase):
    """Permutation pattern: sources and targets (same amount) have 1 connection each, no repetitions"""

    def __init__(self, encoder, n: int = 3):
        super().__init__(encoder, n_src=n, n_tgt=n)

    def get_init_kwargs(self) -> dict:
        return {'n': self._n_src}

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node([1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n={self._n_src})'

    def get_problem_name(self):
        return 'An Perm Prob'

    def __str__(self):
        return f'An Perm Prob {self._n_src} -> {self._n_tgt}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[ManualBestEncoder]:
        return PermutingBestEncoder(imputer)


class PermutingBestEncoder(ManualBestEncoder):
    """Manually encodes the permuting pattern: n design variables with n options [Selva2016],
    and an NSGA2 configured with permutation-specific operators (https://pymoo.org/customization/permutation.html#Flowshop-Schedule)"""

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        return [DiscreteDV(n_opts=self.n_src) for _ in range(self.n_src)]

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[np.ndarray]:
        matrix = np.zeros((self.n_src, self.n_src), dtype=int)
        matrix[np.arange(self.n_src), np.array(vector)] = 1
        return matrix

    def _pattern_name(self) -> str:
        return 'Permuting'

    def get_nsga2(self, pop_size=100, **kwargs) -> Optional[NSGA2]:
        from pymoo.operators.sampling.rnd import PermutationRandomSampling
        from pymoo.operators.crossover.ox import OrderCrossover
        from pymoo.operators.mutation.inversion import InversionMutation
        return NSGA2(
            pop_size=pop_size, eliminate_duplicates=True,
            sampling=PermutationRandomSampling(),
            mutation=InversionMutation(), crossover=OrderCrossover(),
            repair=PermutingRepair(),
            **kwargs,
        )


class PermutingRepair(Repair):
    """Repairs design variables in the permuting problem"""

    def _do(self, problem, X, **kwargs):
        X = np.around(X).astype(int)
        n_perm = X.shape[1]
        for permutation in X:
            _, idx, counts = np.unique(permutation, return_inverse=True, return_counts=True)
            dup_idx = (counts > 1)[idx]
            unique_values = set(permutation[~dup_idx])
            if len(unique_values) == n_perm:
                continue

            new_random_perm = np.random.permutation([i for i in range(n_perm) if i not in unique_values])
            permutation[dup_idx] = new_random_perm

        return X


class AnalyticalIterCombinationsProblem(AnalyticalProblemBase):
    """Unordered non-replacing combining pattern (itertools combinations function: select n_take elements from n_tgt targets):
    1 source has n_take connections to n_tgt targets, no repetition"""

    def __init__(self, encoder, n_take: int = 2, n_tgt: int = 3):
        self._n_take = min(n_take, n_tgt)
        super().__init__(encoder, n_src=1, n_tgt=n_tgt)

    def get_init_kwargs(self) -> dict:
        return {'n_take': self._n_take, 'n_tgt': self._n_tgt}

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node([self._n_take], repeated_allowed=False) if src else Node([0, 1], repeated_allowed=False)

    def __repr__(self):
        return f'{self.__class__.__name__}(n_take={self._n_take}, n_tgt={self._n_tgt})'

    def get_problem_name(self):
        return 'An Iter Comb Prob'

    def __str__(self):
        return f'An Iter Comb Prob {self._n_take} from {self._n_tgt}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[ManualBestEncoder]:
        return UnorderedCombiningBestEncoder(imputer, n_take=self._n_take)


class UnorderedCombiningBestEncoder(ManualBestEncoder):
    """Manually encodes the unordered combining pattern: n_take design variables with n_tgt options, with a repair"""

    def __init__(self, imputer, n_take=2, with_replacement=False):
        self.n_take = n_take
        self.with_replacement = with_replacement
        super().__init__(imputer)

    def _encode(self, existence: NodeExistence) -> List[DiscreteDV]:
        if self.with_replacement:
            return [DiscreteDV(n_opts=self.n_tgt) for _ in range(self.n_take)]

        # Without replacement, each variable can only take n-n_take positions
        return [DiscreteDV(n_opts=self.n_tgt-(self.n_take-1)) for _ in range(self.n_take)]

    def _decode(self, vector: DesignVector, existence: NodeExistence) -> Optional[np.ndarray]:
        matrix = np.zeros((1, self.n_tgt), dtype=int)
        for i_dv, i in enumerate(vector):
            offset = 0 if self.with_replacement else i_dv
            matrix[0, i+offset] += 1
        return matrix

    def _pattern_name(self) -> str:
        return 'Combinations With Replacement' if self.with_replacement else 'Combinations'

    def get_nsga2(self, pop_size=100) -> Optional[NSGA2]:
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        return NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            repair=UnorderedCombiningRepair(self.n_take, self.n_tgt),
        )


class UnorderedCombiningRepair(Repair):
    """Repairs design variables in the unordered combining problem. Works both for with and without replacement, as the version
    without replacement is offset by the index of the variable.
    To repair, it ensures that indices of subsequent design variables are the same or higher than the preceding"""

    def __init__(self, n_take, n):
        self.n_take = n_take
        self.n = n
        super().__init__()

    def _do(self, problem, X, **kwargs):
        X = np.around(X).astype(int)
        for combination in X:
            for i in range(1, self.n_take):
                if combination[i] < combination[i-1]:
                    # Move to any of the random positions that are equal or higher
                    combination[i] = np.random.choice(np.arange(combination[i-1], self.n))
        return X


class AnalyticalIterCombinationsReplacementProblem(AnalyticalIterCombinationsProblem):
    """Unordered combining (with replacements) pattern (itertools combinations_with_replacement function:
    select n_take elements from n_tgt targets):
    1 source has n_take connections to n_tgt targets, repetition allowed"""

    def _get_node(self, src: bool, idx: int) -> Node:
        return Node([self._n_take]) if src else Node(min_conn=0)

    def get_problem_name(self):
        return 'An Iter Comb Repl Prob'

    def __str__(self):
        return f'An Iter Comb Repl Prob {self._n_take} from {self._n_tgt}'

    def get_manual_best_encoder(self, imputer: LazyImputer) -> Optional[ManualBestEncoder]:
        return UnorderedCombiningBestEncoder(imputer, n_take=self._n_take, with_replacement=True)


if __name__ == '__main__':
    from assign_enc.eager.encodings import *
    from assign_enc.eager.imputation import *
    from assign_enc.lazy.encodings import *
    from assign_enc.lazy.imputation import *
    enc = DirectMatrixEncoder(FirstImputer())
    # enc = OneVarEncoder(FirstImputer())
    # enc = LazyDirectMatrixEncoder(LazyFirstImputer())

    # Strange setup like this for profiling
    import timeit

    def _start_comp():
        s = timeit.default_timer()
        p = AnalyticalAssignmentProblem(enc)
        if isinstance(enc, EagerEncoder):
            print(p.assignment_manager.matrix.shape[0])
            # p.eval_points()
        else:
            print(p.get_matrix_count())
        print(f'{timeit.default_timer()-s} sec')

        print(f'Imputation ratio: {p.get_imputation_ratio(n_sample=None)}')
        print(f'Information error: {p.get_information_error()[0, 0]}')

    def _do_real():
        s = timeit.default_timer()
        p = AnalyticalAssignmentProblem(enc, n_src=3, n_tgt=4)
        if isinstance(enc, EagerEncoder):
            print(p.assignment_manager.matrix.shape[0])
            # p.eval_points()
        else:
            print(p.get_matrix_count())
        print(f'{timeit.default_timer()-s} sec')

        print(f'Imputation ratio: {p.get_imputation_ratio(n_sample=None)}')
        print(f'Information error: {p.get_information_error()[0, 0]}')

    # _start_comp()
    # _do_real(), exit()

    from assign_enc.selector import *
    from assign_enc.encoder_registry import *
    from assign_pymoo.metrics_compare import *

    # def _do_time():
    #     EncoderSelector.encoding_timeout = .5
    #     EncoderSelector.n_mat_max_eager = 1e3
    #     def _do_cache(disable_cache):
    #         EncoderSelector._global_disable_cache = disable_cache
    #         s = timeit.default_timer()
    #         # enc = AmountFirstGroupedEncoder(DEFAULT_EAGER_IMPUTER(), SourceAmountFlattenedGrouper(), CoordIndexLocationGrouper())
    #         # p = AnalyticalIterCombinationsProblem(enc, n_take=8, n_tgt=16)
    #         p = AnalyticalIterCombinationsProblem(None, n_take=9, n_tgt=19)
    #         print(f'{p!s}: {p.assignment_manager.encoder!s} ({timeit.default_timer()-s:.1f} sec, '
    #               f'imp_ratio = {p.get_imputation_ratio():.2f}, inf_idx = {p.get_information_index():.2f})')
    #     _do_cache(True)
    #     _do_cache(False)
    # _do_time(), exit()

    # p = AnalyticalCombinationProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER())  # Very high imputation ratios
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER(), injective=True, n_src=2, n_tgt=4)
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER(), injective=True)
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER(), surjective=True)
    # p = AnalyticalAssignmentProblem(DEFAULT_EAGER_ENCODER(), injective=True, surjective=True)
    p = AnalyticalPartitioningProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalPartitioningProblem(DEFAULT_EAGER_ENCODER(), covering=True, n_src=3, n_tgt=4)
    # p = AnalyticalPartitioningProblem(LazyAmountFirstEncoder(DEFAULT_LAZY_IMPUTER(), FlatLazyAmountEncoder(), FlatLazyConnectionEncoder()), n_src=2, n_tgt=4, covering=True)
    # p = AnalyticalDownselectingProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalConnectingProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalPermutingProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalIterCombinationsProblem(DEFAULT_EAGER_ENCODER())
    # p = AnalyticalIterCombinationsReplacementProblem(DEFAULT_EAGER_ENCODER(), n_take=3, n_tgt=3)
    p.reset_pf_cache()
    p.plot_pf(show_approx_f_range=True, n_sample=1000), exit()
    enc = []
    enc += [e(DEFAULT_EAGER_IMPUTER()) for e in EAGER_ENCODERS]
    enc += [e(DEFAULT_EAGER_IMPUTER()) for e in EAGER_ENUM_ENCODERS]
    enc += [e(DEFAULT_LAZY_IMPUTER()) for e in LAZY_ENCODERS]
    # MetricsComparer().compare_encoders(p, enc)
    MetricsComparer().compare_encoders(p, enc, inf_idx=True)
    # MetricsComparer(n_samples=50, n_leave_out=30).check_information_corr(p, enc)
