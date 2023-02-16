import itertools
import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_pymoo.problem import *

__all__ = ['GNCProblem']


class GNCProblem(MultiAssignmentProblem):
    """
    Guidance, Navigation and Control architecture design problem, from chapter 15 of:
    Crawley et al., "System Architecture - Strategy and Product Development for Complex Systems", 2015.

    The challenge is to find the most optimal selection and connection patterns from Sensors to Computers,
    and Computers to Actuators. The number and type of each element can be selected. The architecture is evaluated in
    terms of reliability (more connections and more reliable components lead to more system-level reliability) and
    mass (more reliable components are heavier). This is therefore a multi-objective optimization problem.

    Component mass and probabilities are taken from:
    Apaza & Selva, "Automatic Composition of Encoding Scheme and Search Operators in System Architecture Optimization", 2021.

    This implementation only models the sensor-to-computer connections.
    """

    mass = {
        'S': {'A': 3., 'B': 6., 'C': 9.},
        'C': {'A': 3., 'B': 5., 'C': 10.},
        'A': {'A': 3.5, 'B': 5.5, 'C': 9.5},
    }
    failure_rate = {
        'S': {'A': .00015, 'B': .0001, 'C': .00005},
        'C': {'A': .0001, 'B': .00004, 'C': .00002},
        'A': {'A': .00008, 'B': .0002, 'C': .0001},
    }

    def __init__(self, encoder: Encoder = None, choose_nr=True, n_max=3, choose_type=True, actuators=False):
        self.choose_nr = choose_nr
        self.n_max = n_max
        self.choose_type = choose_type
        self.actuators = actuators
        super().__init__(encoder)

    def get_init_kwargs(self) -> dict:
        return {'choose_nr': self.choose_nr, 'choose_type': self.choose_type, 'n_max': self.n_max,
                'actuators': self.actuators}

    def get_n_obj(self) -> int:
        if not self.choose_nr and not self.choose_type:  # In this case, there is no way to change system mass
            return 1
        return 2

    def get_n_valid_design_points(self, n_cont=5) -> int:

        def _get_n_design_points(n_sensors_, n_computers_, n_actuators_=0) -> int:
            # Get number of connection options
            n_conn = n_comb_conns[n_sensors_, n_computers_]
            if n_actuators_ > 0:
                n_conn *= n_comb_conns[n_computers_, n_actuators_]

            # Type choices and connection choices are independent, so multiply the number of possibilities
            if self.choose_type:
                n_conn *= n_comb_types[n_sensors_]*n_comb_types[n_computers_]
                if n_actuators_ > 0:
                    n_conn *= n_comb_types[n_actuators_]
            return n_conn

        # Get the number of combinations of node connections for each possible number of nodes (symmetric)
        n_node_exist = list(range(1, self.n_max+1)) if self.choose_nr else [self.n_max]
        n_comb_conns = {}
        for n_src, n_tgt in itertools.product(n_node_exist, n_node_exist):
            if n_tgt < n_src:
                continue

            # Only one possible connection pattern possible if one source or target exists (namely connect to all)
            if n_src == 1:
                n_comb_conns[(n_src, n_tgt)] = 1
                n_comb_conns[(n_tgt, n_src)] = 1
            else:
                n_comb_conns[(n_src, n_tgt)] = n_comb = AggregateAssignmentMatrixGenerator.create(
                    src=[Node(min_conn=1, repeated_allowed=False) for _ in range(n_src)],
                    tgt=[Node(min_conn=1, repeated_allowed=False) for _ in range(n_tgt)],
                ).count_all_matrices()
                n_comb_conns[(n_tgt, n_src)] = n_comb

        # Get the number of combinations of node types for each possible number of nodes
        n_comb_types = {}
        if self.choose_type:
            for n_node in n_node_exist:
                n_comb_types[n_node] = len(list(itertools.combinations_with_replacement('ABC', n_node)))

        # Calculate the number of types and connections for each combination of number of nodes
        n_total = 0
        n_act = n_node_exist if self.actuators else [0]
        for n_sen, n_comp, n_act in itertools.product(n_node_exist, n_node_exist, n_act):
            n_total += _get_n_design_points(n_sen, n_comp, n_actuators_=n_act)
        return n_total

    def get_matrix_gen_settings(self) -> List[MatrixGenSettings]:
        matrix_gen_settings = []

        main_conn_patterns = None
        n = 3 if self.actuators else 2
        if self.choose_nr:
            # Choose number by choosing the amount of connections from 1 source to 1 target
            for _ in range(n):  # Sensors, computers, [actuators]
                matrix_gen_settings.append(MatrixGenSettings(
                    src=[Node(list(range(1, self.n_max+1)))], tgt=[Node(min_conn=1)],
                ))

            # Get the different existence patterns for the sensor-computer-actuator connections:
            # the first node always exists, the others are optional
            node_is_conditional = [i > 0 for i in range(self.n_max)]
            main_conn_patterns = NodeExistencePatterns.get_increasing(node_is_conditional, node_is_conditional)

        if self.choose_type:
            # Choose type using a combinations with replacement pattern
            n_nodes_opts = range(1, self.n_max+1) if self.choose_nr else [self.n_max]
            for _ in range(n):  # Sensors, computers, [actuators]
                matrix_gen_settings.append(MatrixGenSettings(
                    # Take n_obj from 3 options (A, B, C)
                    src=[Node([self.n_max])], tgt=[Node(min_conn=0) for _ in range(3)],
                    existence=NodeExistencePatterns([
                        NodeExistence(src_n_conn_override={0: [n]}) for n in n_nodes_opts]),
                ))

        # Main connection assignments
        for _ in range(n-1):
            matrix_gen_settings.append(MatrixGenSettings(
                src=[Node(min_conn=1, repeated_allowed=False) for _ in range(self.n_max)],
                tgt=[Node(min_conn=1, repeated_allowed=False) for _ in range(self.n_max)],
                existence=main_conn_patterns,
            ))

        return matrix_gen_settings

    def _resolve_existence(self, x_parts: List[DesignVector]) -> Tuple[List[DesignVector], List[Optional[List[Tuple[int, int]]]], dict]:
        """Resolve design vectors for each assignment manager into a list of connection lists, or None if any of the
        matrices is invalid"""
        design_vector_parts = []
        connections = []
        assignment_managers = self._assignment_managers
        n = 3 if self.actuators else 2

        def _add_next(dv_, conns_):
            nonlocal x_parts, assignment_managers
            design_vector_parts.append(dv_)
            connections.append(conns_)
            x_parts = x_parts[1:]
            assignment_managers = assignment_managers[1:]

        # Get number of chosen objects (sensors, computers, actuators) from the first assignment managers if we choose the numbers
        if self.choose_nr:
            n_objs = []
            for i in range(n):
                dv, conns = assignment_managers[0].get_conn_idx(x_parts[0])
                _add_next(dv, conns)
                n_objs.append(len(conns) if conns is not None else self.n_max)

        # If numbers are not chosen, set object amounts to max nr
        else:
            n_objs = [self.n_max for _ in range(n)]

        # Choose object types from subsequent assignment managers
        if self.choose_type:
            for n_obj in n_objs:
                obj_existence = NodeExistence(src_n_conn_override={0: [n_obj]})
                dv, conns = assignment_managers[0].get_conn_idx(x_parts[0], existence=obj_existence)
                _add_next(dv, conns)

        # Get inter-object connections
        for i_conns, n_src in enumerate(n_objs[:-1]):
            n_tgt = n_objs[i_conns+1]
            conn_existence = NodeExistence(src_exists=[i < n_src for i in range(self.n_max)],
                                           tgt_exists=[i < n_tgt for i in range(self.n_max)])

            dv, conns = assignment_managers[0].get_conn_idx(x_parts[0], existence=conn_existence)
            _add_next(dv, conns)

        eval_kwargs = {'n_objs': n_objs}
        return design_vector_parts, connections, eval_kwargs

    def _do_evaluate(self, conn_list: List[List[Tuple[int, int]]], **eval_kwargs) -> Tuple[List[float], List[float]]:
        n_objs: List[int] = eval_kwargs['n_objs']

        if self.choose_nr:
            conn_list = conn_list[len(n_objs):]

        # Get object types from connections list (or by cycling available types of not choosing types)
        types = ['A', 'B', 'C']
        obj_types = []
        if self.choose_type:
            for _ in n_objs:
                obj_types.append([types[i_tgt] for _, i_tgt in conn_list[0]])
                conn_list = conn_list[1:]

        else:
            for n in n_objs:
                type_cycle = itertools.cycle(types)
                obj_types.append([next(type_cycle) for _ in range(n)])

        assert len(obj_types) == len(n_objs) == (3 if self.actuators else 2)

        # Calculate metrics (both to be minimized)
        mass = self._calc_mass(obj_types[0], obj_types[1], actuator_types=obj_types[2] if self.actuators else None)
        failure_rate = self._calc_failure_rate(
            obj_types[0], obj_types[1], conn_list[0],
            actuator_types=obj_types[2] if self.actuators else None, act_conns=conn_list[1] if self.actuators else None)

        if not self.choose_nr and not self.choose_type:
            return [failure_rate], []
        return [failure_rate, mass], []

    @classmethod
    def _calc_mass(cls, sensor_types, computer_types, actuator_types=None):
        mass = sum([cls.mass['S'][type_] for type_ in sensor_types])
        mass += sum([cls.mass['C'][type_] for type_ in computer_types])
        if actuator_types is not None:
            mass += sum([cls.mass['A'][type_] for type_ in actuator_types])
        return mass

    @classmethod
    def _calc_failure_rate(cls, sensor_types, computer_types, conns, actuator_types=None, act_conns=None):

        # Get item failure rates
        rate = cls.failure_rate
        failure_rates = [np.array([rate['S'][type_] for type_ in sensor_types]),
                         np.array([rate['C'][type_] for type_ in computer_types])]
        obj_conns = [conns]
        if actuator_types is not None:
            failure_rates.append(np.array([rate['A'][type_] for type_ in actuator_types]))
            obj_conns.append(act_conns)

        conn_matrices = []
        for i, edges in enumerate(obj_conns):
            matrix = np.zeros((len(failure_rates[i]), len(failure_rates[i+1])), dtype=int)
            for i_src, i_tgt in edges:
                matrix[i_src, i_tgt] = 1
            conn_matrices.append(matrix)

        # Loop over combinations of failed components
        def _branch_failures(i_rates=0, src_connected_mask=None) -> float:
            calc_downstream = i_rates < len(conn_matrices)-1
            rates, tgt_rates = failure_rates[i_rates], failure_rates[i_rates+1]
            conn_mat = conn_matrices[i_rates]

            # Loop over failure scenarios
            if src_connected_mask is None:
                src_connected_mask = np.ones((len(rates),), dtype=bool)
            total_rate = 0.
            for ok_sources in itertools.product(*[([False, True] if src_connected_mask[i_conn] else [False]) for i_conn in range(len(rates))]):
                if i_rates > 0 and not any(ok_sources):
                    continue

                # Calculate probability of this scenario occurring
                ok_sources = list(ok_sources)
                occurrence_prob = rates.copy()
                occurrence_prob[ok_sources] = 1-occurrence_prob[ok_sources]
                prob = 1.
                for partial_prob in occurrence_prob[src_connected_mask]:
                    prob *= partial_prob
                occurrence_prob = prob

                # Check which targets are still connected in this scenario
                conn_mat_ok = conn_mat[ok_sources, :].T
                connected_targets = np.zeros((conn_mat_ok.shape[0],), dtype=bool)
                for i_conn_tgt in range(conn_mat_ok.shape[0]):
                    connected_targets[i_conn_tgt] = np.any(conn_mat_ok[i_conn_tgt])

                # If no connected targets are available the system fails
                tgt_failure_rates = tgt_rates[connected_targets]
                if len(tgt_failure_rates) == 0:
                    total_rate += occurrence_prob
                    continue

                # Calculate the probability that the system fails because all remaining connected targets fail
                all_tgt_fail_prob = 1.
                for prob in tgt_failure_rates:
                    all_tgt_fail_prob *= prob
                total_rate += occurrence_prob*all_tgt_fail_prob

                # Calculate the probability that the system fails because remaining downstream connected targets fail
                if calc_downstream:
                    total_rate += occurrence_prob*_branch_failures(i_rates=i_rates+1, src_connected_mask=connected_targets)

            return total_rate

        failure_rate = _branch_failures()
        return np.log10(failure_rate)

    def __repr__(self):
        return f'{self.__class__.__name__}(choose_nr={self.choose_nr}, n_max={self.n_max}, ' \
               f'choose_type={self.choose_type}, actuators={self.actuators})'

    def get_problem_name(self):
        features = []
        if self.choose_nr:
            features.append('NR')
        if self.choose_type:
            features.append('TYP')
        if self.actuators:
            features.append('ACT')
        return f'GNC {"/".join(features)}'.strip()

    def __str__(self):
        return f'{self.get_problem_name()} @ {self.n_max}'


if __name__ == '__main__':
    from assign_enc.encoder_registry import *
    from assign_pymoo.metrics_compare import *
    nm = 3
    # p = GNCProblem(DEFAULT_EAGER_ENCODER(), choose_nr=False, n_max=nm, choose_type=False)
    # p = GNCProblem(DEFAULT_EAGER_ENCODER(), choose_nr=False, n_max=nm, choose_type=True)
    # p = GNCProblem(DEFAULT_EAGER_ENCODER(), choose_nr=True, n_max=nm, choose_type=False)
    # p = GNCProblem(DEFAULT_EAGER_ENCODER(), choose_nr=True, n_max=nm, choose_type=True)

    # p = GNCProblem(DEFAULT_EAGER_ENCODER(), choose_nr=False, n_max=nm, choose_type=False, actuators=True)
    # p = GNCProblem(DEFAULT_EAGER_ENCODER(), choose_nr=False, n_max=nm, choose_type=True, actuators=True)
    # p = GNCProblem(DEFAULT_EAGER_ENCODER(), choose_nr=True, n_max=nm, choose_type=False, actuators=True)
    p = GNCProblem(DEFAULT_EAGER_ENCODER(), choose_nr=True, n_max=nm, choose_type=True, actuators=True)

    # p = GNCProblem(DEFAULT_LAZY_ENCODER(), choose_nr=False, n_max=nm, choose_type=False)
    # p = GNCProblem(DEFAULT_LAZY_ENCODER(), choose_nr=False, n_max=nm, choose_type=True)
    # p = GNCProblem(DEFAULT_LAZY_ENCODER(), choose_nr=True, n_max=nm, choose_type=False)
    # p = GNCProblem(DEFAULT_LAZY_ENCODER(), choose_nr=True, n_max=nm, choose_type=True)

    print(f'Design space size: {p.get_n_design_points()}')
    print(f'Valid designs: {p.get_n_valid_design_points()}')
    print(f'Imputation ratio: {p.get_imputation_ratio():.2f}')
    # p.get_n_valid_design_points = lambda **_: None
    # print(f'Imputation ratio: {p.get_imputation_ratio():.2f}')
    # exit()
    # p.plot_points(n=5000), exit()

    # from pymoo.core.evaluator import Evaluator
    # from assign_pymoo.sampling import RepairedRandomSampling
    # RepairedRandomSampling(repair=p.get_repair()).do(p, 1)
    # def _wrapped():
    #     Evaluator().eval(p, RepairedRandomSampling(repair=p.get_repair()).do(p, 100))
    # _wrapped(), exit()

    # p.reset_pf_cache()
    p.plot_pf(show_approx_f_range=True, n_sample=1000), exit()

    enc = []
    enc += [e(DEFAULT_EAGER_IMPUTER()) for e in EAGER_ENCODERS]
    enc += [e(DEFAULT_LAZY_IMPUTER()) for e in LAZY_ENCODERS]
    MetricsComparer().compare_encoders(p, enc)
