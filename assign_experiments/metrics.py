import numpy as np
from typing import *
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

from pymoo.indicators.igd import IGD
from pymoo.core.algorithm import Algorithm
from pymoo.core.indicator import Indicator
from pymoo.indicators.hv import Hypervolume
from pymoo.core.population import Population
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.core.termination import Termination
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.indicators.distance_indicator import euclidean_distance
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

__all__ = ['Metric', 'IndicatorMetric', 'MetricTermination', 'MetricDiffTermination', 'SpreadMetric',
           'DeltaHVMetric', 'IGDMetric', 'IGDPlusMetric', 'MaxConstraintViolationMetric', 'NrEvaluationsMetric',
           'BestObjMetric']


class Metric:
    """More general metric class, that can output multiple values to track. Should be serializable."""

    def __init__(self):
        self.values = {name: [] for name in self.value_names}
        self.values_std = None  # Used in ExperimenterResults

    def calculate_step(self, algorithm: Algorithm):
        values = self._calculate_values(algorithm)
        names = self.value_names
        if len(values) != len(names):
            raise ValueError('Values should have the same length as the number of values')

        for i, name in enumerate(names):
            self.values[name].append(values[i])

    def results(self) -> Dict[str, np.ndarray]:
        return {key: np.array(value) for key, value in self.values.items()}

    def results_std(self) -> Dict[str, np.ndarray]:
        if self.values_std is None:
            return {}
        return {key: np.array(value) for key, value in self.values_std.items()}

    def plot(self, std_sigma=1., show=True, **kwargs):
        self.plot_multiple([self], std_sigma=std_sigma, show=show, **kwargs)

    @staticmethod
    def plot_multiple(metrics: List['Metric'], titles: List[str] = None, colors: List[str] = None,
                      plot_value_names: List[str] = None, std_sigma=1., n_eval: List[List[float]] = None,
                      save_filename=None, show=True):
        """Function for plotting multiple metrics of the same kind, but coming from different optimization runs."""

        type_ = type(metrics[0])
        if not all([isinstance(m, type_) for m in metrics]):
            raise ValueError('Metrics should be of same type!')

        if colors is not None and len(colors) != len(metrics):
            raise ValueError('Provide same amount of colors as metrics!')

        if titles is not None and len(titles) != len(metrics):
            raise ValueError('Provide same amount of titles as metrics!')

        if plot_value_names is None:
            plot_value_names = metrics[0].value_names

        style = ['-', '-.', ':']
        for value_name in plot_value_names:
            plt.figure(figsize=(16, 12))

            x_max = None
            err_title = ''
            for i, metric in enumerate(metrics):
                y = np.atleast_1d(metric.values[value_name])

                if n_eval is not None:
                    x = np.atleast_1d(n_eval[i])
                    if len(x) != len(y):
                        raise ValueError('List with evaluations should be same length as number of steps!')
                else:
                    x = list(range(len(y)))

                y_err = np.array(metric.values_std[value_name]) if metric.values_std is not None else None

                kwargs = {'linewidth': 1}
                if len(metrics) == 1:
                    kwargs['color'] = 'k'
                elif colors is not None:
                    kwargs['color'] = colors[i]

                line_style = '-'
                if colors is None:
                    line_style = style[i // 10]

                if titles is not None:
                    kwargs['label'] = titles[i]

                l, = plt.plot(x, y, line_style, **kwargs)
                color = l.get_color()
                kwargs['color'] = color

                if y_err is not None and std_sigma != 0.:
                    if 'label' in kwargs:
                        del kwargs['label']

                    err_title = ' (std $\\sigma$ = %.2f)' % std_sigma
                    plt.errorbar(x, y+y_err*std_sigma, fmt='--', **kwargs)
                    plt.errorbar(x, y-y_err*std_sigma, fmt='--', **kwargs)

                metric.plot_fig_callback(x, value_name, color=None if len(metrics) == 1 else color)

                if x_max is None or x[-1] > x_max:
                    x_max = x[-1]

            plt.title('Metric: %s.%s%s' % (metrics[0].name, value_name, err_title))
            plt.xlim([0, x_max])
            plt.xlabel('Iterations' if n_eval is None else 'Function evaluations')
            plt.ylabel(value_name)

            if titles is not None:
                plt.legend()

            if save_filename is not None:
                save_value_filename = '%s_%s' % (save_filename, secure_filename(value_name))
                plt.savefig(save_value_filename+'.png')
                plt.savefig(save_value_filename+'.svg')

        if show:
            plt.show()

    @classmethod
    def calc_doe(cls, problem, n_samples=10000, sampling=None, output=True):
        from pymoo.core.evaluator import Evaluator
        if sampling is None:
            from pymoo.operators.sampling.rnd import FloatRandomSampling
            sampling = FloatRandomSampling()

        pop = sampling.do(problem, n_samples=n_samples)
        Evaluator().eval(problem, pop)

        dummy_algo = Algorithm()
        dummy_algo.pop = pop
        dummy_algo.advance()
        dummy_algo._set_optimum()

        metric = cls()
        metric.calculate_step(dummy_algo)

        if output:
            print('%s results for %r' % (cls.__name__, problem))
            for value_name in metric.value_names:
                print('%s: %r' % (value_name, metric.values[value_name][-1]))
        return metric.results()

    def plot_fig_callback(self, x, value_name: str, color=None):
        pass

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def value_names(self) -> List[str]:
        raise NotImplementedError

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        raise NotImplementedError

    @classmethod
    def _get_pop_x(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Design vectors of the population: (n_pop, n_x)"""
        return cls._get_pop(algorithm, feasible_only=feasible_only).get('X')

    @classmethod
    def _get_pop_f(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Objective values of the population: (n_pop, n_f)"""
        return cls._get_pop(algorithm, feasible_only=feasible_only).get('F')

    @classmethod
    def _get_pop_g(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Constraint values of the population: (n_pop, n_g)"""
        return cls._get_pop(algorithm, feasible_only=feasible_only).get('G')

    @classmethod
    def _get_pop_cv(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Constraint violation values of the population: (n_pop, n_g)"""
        return cls._get_pop(algorithm, feasible_only=feasible_only).get('CV')

    @staticmethod
    def _get_pop(algorithm: Algorithm, feasible_only=False):
        pop = algorithm.pop
        if feasible_only:
            i_feasible = np.where(pop.get('feasible'))[0]
            return pop[i_feasible]
        return pop

    @classmethod
    def _get_opt_x(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Design vectors of the optimum population (non-dominated current Pareto front): (n_opt, n_x)"""
        return cls._get_opt(algorithm, feasible_only=feasible_only).get('X').astype(np.float)

    @classmethod
    def _get_opt_f(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Objective values of the optimum population: (n_opt, n_f)"""
        return cls._get_opt(algorithm, feasible_only=feasible_only).get('F').astype(np.float)

    @classmethod
    def _get_opt_g(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Constraint values of the optimum population: (n_opt, n_g)"""
        return cls._get_opt(algorithm, feasible_only=feasible_only).get('G').astype(np.float)

    @classmethod
    def _get_opt_cv(cls, algorithm: Algorithm, feasible_only=False) -> np.ndarray:
        """Constraint violation values of the optimum population: (n_opt, n_g)"""
        return cls._get_opt(algorithm, feasible_only=feasible_only).get('CV').astype(np.float)

    @staticmethod
    def _get_opt(algorithm: Algorithm, feasible_only=False):
        opt = algorithm.opt
        if feasible_only:
            i_feasible = np.where(opt.get('feasible'))[0]
            return opt[i_feasible]
        return opt

    @staticmethod
    def get_pareto_front(f: np.ndarray) -> np.ndarray:
        """Get the non-dominated set of objective values (the Pareto front)."""
        i_non_dom = NonDominatedSorting().do(f, only_non_dominated_front=True)
        return np.copy(f[i_non_dom, :])


class IndicatorMetric(Metric):
    """Metric based on a performance indicator. Performance indicators only depend on the objective values."""

    def __init__(self, indicator: Indicator):
        super(IndicatorMetric, self).__init__()
        self.indicator = indicator

    @property
    def name(self) -> str:
        return self.indicator.__class__.__name__

    @property
    def value_names(self) -> List[str]:
        return ['indicator']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        return [self.indicator(self._get_opt_f(algorithm))]


class MetricTermination(Termination):
    """Termination based on a metric."""

    def __init__(self, metric: Metric, value_name: str = None, lower_limit: float = None, upper_limit: float = None,
                 n_eval_check: int = None):
        if lower_limit is None and upper_limit is None:
            raise ValueError('Provide at least either a lower or an upper limit!')
        self.metric = metric
        self.n_eval = []
        self.value_name = value_name or metric.value_names[0]
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.n_eval_check = n_eval_check

        super(MetricTermination, self).__init__()

    @property
    def metric_name(self) -> str:
        return self.metric.name

    def _do_continue(self, algorithm: Algorithm, **kwargs):

        values = self._calc_step(algorithm)
        value = values[-1]

        if self.lower_limit is not None and value <= self.lower_limit:
            return False
        if self.upper_limit is not None and value >= self.upper_limit:
            return False
        return True

    def _calc_step(self, algorithm: Algorithm):
        do_calc = True
        if len(self.n_eval) > 0 and self.n_eval_check is not None:
            n_eval_next = self.n_eval[-1]+self.n_eval_check
            n_eval_algo = algorithm.evaluator.n_eval
            if n_eval_algo < n_eval_next:
                do_calc = False

        if do_calc:
            self.metric.calculate_step(algorithm)
            self.n_eval.append(algorithm.evaluator.n_eval)
        return self._get_check_values()

    def _get_check_values(self):
        return np.array(self.metric.values[self.value_name])

    def plot(self, save_filename=None, show=True):
        plt.figure()
        plt.title('Metric termination: %s.%s' % (self.metric_name, self.value_name))

        y = self.metric.values[self.value_name]
        x = list(range(len(y)))

        plt.plot(x, y, '-k', linewidth=1)
        if self.lower_limit is not None:
            plt.plot(x, np.ones((len(x),))*self.lower_limit, '--k', linewidth=1)
        if self.upper_limit is not None:
            plt.plot(x, np.ones((len(x),))*self.upper_limit, '--k', linewidth=1)

        plt.xlim([0, x[-1]])
        plt.xlabel('Iteration')
        plt.ylabel(self.value_name)

        if save_filename is not None:
            plt.savefig(save_filename+'.png')
            # plt.savefig(save_filename+'.svg')
        if show:
            plt.show()


class MetricDiffTermination(MetricTermination):
    """Termination based on the rate of change of a metric."""

    def __init__(self, metric: Metric, value_name: str = None, limit: float = None, **kwargs):
        super(MetricDiffTermination, self).__init__(metric, value_name=value_name, lower_limit=limit, **kwargs)

        self.diff_values = []

    def _do_continue(self, algorithm: Algorithm, **kwargs):

        values = self._calc_step(algorithm)
        values = np.array(values)
        real_values = values[~np.isnan(values)]

        if len(real_values) < 2:
            self.diff_values.append(np.nan)
            return True

        diff = abs(real_values[-1]-real_values[-2])
        self.diff_values.append(diff)
        return diff > self.lower_limit

    def plot(self, save_filename=None, show=True):
        _ll = self.lower_limit
        self.lower_limit = None
        super(MetricDiffTermination, self).plot(save_filename=save_filename, show=False)
        self.lower_limit = _ll

        plt.figure()
        plt.title('Metric termination (diff): %s.%s' % (self.metric_name, self.value_name))

        y = self.diff_values
        x = list(range(len(y)))

        plt.semilogy(x, y, '-k', linewidth=1)
        plt.semilogy(x, np.ones((len(x),))*self.lower_limit, '--k', linewidth=1)

        plt.xlim([0, x[-1]])
        plt.xlabel('Iteration')
        plt.ylabel(self.value_name+' diff')

        if save_filename is not None:
            plt.savefig(save_filename+'_diff.png')
            # plt.savefig(save_filename+'_diff.svg')
        if show:
            plt.show()


class SpreadMetric(Metric):
    """
    Spread measures how well-spread a Pareto front is, representing the exploration performance of the algorithm. This
    metric only works for problems with 2 objectives. A value of 0 indicates a perfectly uniform spread.

    Implementation based on:
    Deb, K., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II", 2002, 10.1109/4235.996017
    """

    @property
    def name(self) -> str:
        return 'spread'

    @property
    def value_names(self) -> List[str]:
        return ['delta']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        if algorithm.problem.n_obj != 2:
            raise ValueError('Spread metric is only available for problems with 2 objectives!')

        # Get objective values of the current Pareto front (n_opt, n_obj), and sort along the first objective
        f = self._get_opt_f(algorithm)
        f = f[np.argsort(f[:, 0]), :]

        if f.shape[0] < 3:
            return [1.]

        dists = euclidean_distance(f[:-1, :], f[1:, :], norm=1)
        extreme_dists = dists[0]+dists[-1]  # d_f + d_i

        internal_dists = dists[1:-1]
        d_mean = np.mean(internal_dists)
        n_internal = len(internal_dists)

        # Equation (1), page 7 (188)
        delta = (extreme_dists + np.sum(np.abs(internal_dists - d_mean))) /\
                (extreme_dists + n_internal*d_mean)
        return [delta]


class DeltaHVMetric(Metric):
    """
    Metric measuring the difference to the pre-known hypervolume. It has a value between 1 and 0, where 0 means the
    hypervolume is exactly the same, meaning the true Pareto front has been found.

    Implementation based on:
    Palar, P.S., "On Multi-Objective Efficient Global Optimization Via Universal Kriging Surrogate Model", 2017,
        10.1109/CEC.2017.7969368
    """

    def __init__(self, pf: np.ndarray):
        super(DeltaHVMetric, self).__init__()

        self._hv = hv = Hypervolume(pf=pf, normalize=True)
        self.hv_true = hv(pf)

    @property
    def name(self) -> str:
        return 'delta_hv'

    @property
    def value_names(self) -> List[str]:
        return ['delta_hv', 'hv', 'true_hv']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        # Calculate current hypervolume
        f = self._get_pop_f(algorithm)
        hv = self._hv(f)

        # Calculate error metric
        delta_hv = (self.hv_true-hv)/self.hv_true

        return [delta_hv, hv, self.hv_true]


class IGDMetric(IndicatorMetric):
    """Inverse generational distance to the known pareto front."""

    def __init__(self, pf):
        super(IGDMetric, self).__init__(IGD(pf, normalize=True))


class IGDPlusMetric(IndicatorMetric):
    """Inverse generational distance (improved) to the known pareto front."""

    def __init__(self, pf):
        super(IGDPlusMetric, self).__init__(IGDPlus(pf, normalize=True))


class MaxConstraintViolationMetric(Metric):
    """Metric that simply returns the maximum constraint violation of the current population."""

    def __init__(self):
        super(MaxConstraintViolationMetric, self).__init__()

        self._total_pop = None
        self._el_dup = DefaultDuplicateElimination()

    @property
    def name(self) -> str:
        return 'max_cv'

    @property
    def value_names(self) -> List[str]:
        return ['max_cv', 'min_cv', 'pop_max_cv', 'pop_min_cv', 'frac_nan']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        if self._total_pop is None:
            self._total_pop = self._get_pop(algorithm)
        else:
            pop = Population.merge(self._total_pop, self._get_pop(algorithm))
            self._total_pop = self._el_dup.do(pop)

        cv = self._get_opt_cv(algorithm)
        if len(cv) == 0:
            return [0., 0., 0., 0., 0.]
        cv[np.isinf(cv)] = np.nan

        cv_pop = self._get_pop_cv(algorithm)
        cv_pop[np.isinf(cv_pop)] = np.nan

        cv_total_pop = self._total_pop.get('CV')
        cv_total_pop[np.isinf(cv_total_pop)] = np.nan
        frac_nan = np.sum(np.isnan(cv_total_pop))/len(cv_total_pop)

        return [np.nanmax(cv), np.nanmin(cv), np.nanmax(cv_pop), np.nanmin(cv_pop), frac_nan]


class NrEvaluationsMetric(Metric):
    """Metric that tracks the number of function evaluations after each algorithm step."""

    @property
    def name(self) -> str:
        return 'n_eval'

    @property
    def value_names(self) -> List[str]:
        return ['n_eval']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        return [algorithm.evaluator.n_eval]


class BestObjMetric(Metric):
    """Metric that tracks the current best (feasible) objective values."""

    def __init__(self, i_f=0):
        self.i_f = i_f
        super(BestObjMetric, self).__init__()

    @property
    def name(self):
        return 'f_best'

    @property
    def value_names(self) -> List[str]:
        return ['f_best']

    def _calculate_values(self, algorithm: Algorithm) -> List[float]:
        if algorithm.opt is not None:
            return [algorithm.opt.get('F')[self.i_f, 0]]
        return [np.nan]
