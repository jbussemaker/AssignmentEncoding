import itertools
import numpy as np
from typing import *
from assign_enc.matrix import *
from assign_enc.encoding import *
from assign_pymoo.problem import *
from assign_enc.assignment_manager import AssignmentManager

__all__ = ['GNCProblem']


class GNCProblem(AssignmentProblem):
    """
    Guidance, Navigation and Control architecture design problem, from chapter 15 of:
    Crawley et al., "System Architecture - Strategy and Product Development for Complex Systems", 2015.

    The challenge is to find the most optimal selection and connection patterns from Sensors to Computers,
    and Computers to Actuators. The number and type of each element can be selected. The architecture is evaluated in
    terms of reliability (more connections and more reliable components lead to more system-level reliability) and
    mass (more reliable components are heavier). This is therefore a multi-objective optimization problem.

    Component mass and probabilities are approximated from the original text.

    This implementation only models the sensor-to-computer connections.
    """

    mass = {
        'C': {'A': 5., 'B': 10., 'C': 15.},
        'S': {'A': 1., 'B': 2., 'C': 6.},
    }
    failure_rate = {
        'C': {'A': .004, 'B': .0025, 'C': .001},
        'S': {'A': .001, 'B': .0005, 'C': .0001},
    }

    def __init__(self, encoder: Encoder, choose_nr=True, n_max=3, choose_type=True):
        self.choose_nr = choose_nr
        self.n_max = n_max
        self.choose_type = choose_type
        self._choose_type_manager: Optional[AssignmentManager] = None
        super().__init__(encoder)

    def get_init_kwargs(self) -> dict:
        return {'choose_nr': self.choose_nr, 'choose_type': self.choose_type}

    def get_n_obj(self) -> int:
        return 2

    def get_aux_des_vars(self) -> Optional[List[DiscreteDV]]:
        aux_des_vars = []

        # Design variables for choosing the number of sensors and computers
        aux_node_existence_pattern = None
        if self.choose_nr:
            aux_des_vars += [
                DiscreteDV(n_opts=self.n_max),  # Nr of sensors
                DiscreteDV(n_opts=self.n_max),  # Nr of computers
            ]

            aux_node_existence_pattern = NodeExistencePatterns.get_increasing(
                src_is_conditional=[False], tgt_is_conditional=[i > 0 for i in range(self.n_max)])

        # Design variables for choosing the type of each element: A, B, C
        # Here, we cannot just give three choices per element, because this would lead to architecture redundancies:
        # the selection AAB is logically the same as ABA (itertools.combinations_with_replacement would give all
        # possibilities). For AnalyticalIterCombinationsReplacementProblem (which is equivalent to
        # combinations_with_replacement), the most informative way to encode without any information loss, is the
        # ElementGroupedEncoder encoder.
        if self.choose_type:
            from assign_enc.eager.imputation.closest import ClosestImputer
            from assign_enc.eager.encodings.group_element import ElementGroupedEncoder
            self._choose_type_manager = manager = AssignmentManager(
                src=[Node([self.n_max])],  # Make n_max connections
                # To n_max nodes, repetitions allowed
                tgt=[Node(min_conn=0) for _ in range(self.n_max)],
                existence_patterns=aux_node_existence_pattern,
                encoder=ElementGroupedEncoder(ClosestImputer()))

            aux_des_vars += [DiscreteDV(n_opts=dv.n_opts) for i, dv in enumerate(manager.design_vars)]  # Sensor types
            aux_des_vars += [DiscreteDV(n_opts=dv.n_opts) for i, dv in enumerate(manager.design_vars)]  # Computer types

        return aux_des_vars

    def get_src_tgt_nodes(self) -> Tuple[List[Node], List[Node]]:
        cond, n = self.choose_nr, self.n_max
        sensor_conn = [Node(min_conn=1, repeated_allowed=False) for _ in range(n)]
        comp_conn = [Node(min_conn=1, repeated_allowed=False) for _ in range(n)]
        return sensor_conn, comp_conn

    def get_existence_patterns(self) -> Optional[NodeExistencePatterns]:
        if not self.choose_nr:
            return

        src_is_conditional = [i > 0 for i in range(self.n_max)]
        tgt_is_conditional = [i > 0 for i in range(self.n_max)]
        return NodeExistencePatterns.get_increasing(src_is_conditional, tgt_is_conditional)

    def correct_x_aux(self, x_aux: DesignVector) -> Tuple[DesignVector, Optional[NodeExistence]]:
        # Determine which target nodes exist
        choose_type, n = self.choose_type, self.n_max
        if self.choose_nr:
            n_src, n_tgt = x_aux[0]+1, x_aux[1]+1
            existence = NodeExistence(src_exists=[i < n_src for i in range(n)],
                                      tgt_exists=[i < n_tgt for i in range(n)])
            src_existence = NodeExistence(tgt_exists=existence.src_exists)
            tgt_existence = NodeExistence(tgt_exists=existence.tgt_exists)
            x_nr = x_aux[:2]
        else:
            existence = src_existence = tgt_existence = None
            x_nr = []

        # Correct type selection design variables
        if self.choose_type:
            i0 = 2 if self.choose_nr else 0
            manager = self._choose_type_manager
            n_dv = len(manager.design_vars)

            x_type_src = x_aux[i0:i0+n_dv]
            x_type_src = manager.correct_vector(x_type_src, existence=src_existence)

            x_type_tgt = x_aux[i0+n_dv:]
            x_type_tgt = manager.correct_vector(x_type_tgt, existence=tgt_existence)
        else:
            x_type_src = x_type_tgt = []

        x_corrected = list(x_nr)+list(x_type_src)+list(x_type_tgt)
        return x_corrected, existence

    def _do_evaluate(self, conns: List[Tuple[int, int]], x_aux: Optional[DesignVector]) -> Tuple[List[float], List[float]]:
        # Get number of sensors and computers
        choose_type, n = self.choose_type, self.n_max
        if self.choose_nr:
            n_sensors, n_computers = x_aux[0]+1, x_aux[1]+1
            src_existence = NodeExistence(tgt_exists=[i < n_sensors for i in range(n)])
            tgt_existence = NodeExistence(tgt_exists=[i < n_computers for i in range(n)])

            src_idx = {i for i, _ in conns}
            assert len(src_idx) == n_sensors
            tgt_idx = {i for _, i in conns}
            assert len(tgt_idx) == n_computers
        else:
            n_sensors = n_computers = self.n_max
            src_existence = tgt_existence = None

        # Get sensor and computer types
        types = ['A', 'B', 'C']
        if self.choose_type:
            i0 = 2 if self.choose_nr else 0
            manager = self._choose_type_manager
            n_dv = len(manager.design_vars)

            _, conn_idx = manager.get_conn_idx(x_aux[i0:i0+n_dv], existence=src_existence)
            sensor_types = sorted([types[i_type] for _, i_type in conn_idx])[:n_sensors]

            _, conn_idx = manager.get_conn_idx(x_aux[i0+n_dv:], existence=tgt_existence)
            comp_types = sorted([types[i_type] for _, i_type in conn_idx])[:n_computers]
        else:
            sensor_types = (types*int(np.ceil(n_sensors/len(types))))[:n_sensors]
            comp_types = (types*int(np.ceil(n_computers/len(types))))[:n_computers]

        # Calculate metrics (both to be minimized)
        mass = self._calc_mass(sensor_types, comp_types)
        failure_rate = self._calc_failure_rate(sensor_types, comp_types, conns)

        return [failure_rate, mass], []

    @classmethod
    def _calc_mass(cls, sensor_types, computer_types):
        sensor_mass = sum([cls.mass['S'][type_] for type_ in sensor_types])
        computer_mass = sum([cls.mass['C'][type_] for type_ in computer_types])
        return sensor_mass+computer_mass

    @classmethod
    def _calc_failure_rate(cls, sensor_types, computer_types, conns):

        def system_fails(sensor_failed, computer_failed):
            # Find remaining connections for the failed sensors and computers
            remaining_conns = [conn for conn in conns if not sensor_failed[conn[0]] and not computer_failed[conn[1]]]

            # If there are no remaining connections, the system has failed
            return len(remaining_conns) == 0

        # Get item failure rates
        rate = cls.failure_rate
        failure_rates = [rate['S'][type_] for type_ in sensor_types] + [rate['C'][type_] for type_ in computer_types]

        # Loop over number of failures
        failure_rate = 0.
        n_s = len(sensor_types)
        n_c = len(computer_types)
        n_obj = n_s+n_c
        for n_failed in range(n_obj):
            base_failures = [i < n_failed+1 for i in range(n_obj)]

            # Iterate over possible failure permutations
            for failure_scheme in set(itertools.permutations(base_failures)):
                s_failed = failure_scheme[:n_s]
                c_failed = failure_scheme[n_s:]

                # Check if system fails
                if system_fails(s_failed, c_failed):
                    # Determine probability of this state happening
                    prob = 1.
                    for i, f_rate in enumerate(failure_rates):
                        if failure_scheme[i]:
                            prob *= f_rate

                    failure_rate += prob

        return np.log10(failure_rate)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._encoder!r}, choose_nr={self.choose_nr}, n_max={self.n_max}, ' \
               f'choose_type={self.choose_type})'

    def __str__(self):
        features = []
        if self.choose_nr:
            features.append('NR')
        if self.choose_type:
            features.append('TYP')
        return f'GNC {self.n_max} {"/".join(features)}'.strip()


if __name__ == '__main__':
    from assign_experiments.encoders import *
    from assign_pymoo.metrics_compare import *
    # p = GNCProblem(DEFAULT_EAGER_ENCODER(), choose_nr=False, n_max=3, choose_type=False)
    # p = GNCProblem(DEFAULT_EAGER_ENCODER(), choose_nr=False, n_max=3, choose_type=True)
    p = GNCProblem(DEFAULT_EAGER_ENCODER(), choose_nr=True, n_max=3, choose_type=False)
    # p = GNCProblem(DEFAULT_EAGER_ENCODER(), choose_nr=True, n_max=3, choose_type=True)

    print(f'Design space size: {p.get_n_design_points()}')
    print(f'Imputation ratio: {p.get_imputation_ratio():.2f}')
    p.plot_points(n=5000), exit()

    enc = []
    enc += [e(DEFAULT_EAGER_IMPUTER()) for e in EAGER_ENCODERS]
    enc += [e(DEFAULT_LAZY_IMPUTER()) for e in LAZY_ENCODERS]
    MetricsComparer().compare_encoders(p, enc)
