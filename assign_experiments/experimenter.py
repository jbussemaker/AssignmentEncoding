import os
import bz2
import copy
import pickle
import logging
import tempfile
import contextlib
import numpy as np
import pandas as pd
from typing import *
import logging.config
import concurrent.futures
import matplotlib.pyplot as plt
import pylab as pl

from assign_experiments.metrics import *
from werkzeug.utils import secure_filename

from pymoo.optimize import minimize
from pymoo.core.result import Result
from pymoo.core.problem import Problem
from pymoo.core.algorithm import Algorithm
from pymoo.termination.max_eval import MaximumFunctionCallTermination

__all__ = ['Experimenter', 'EffectivenessTerminator', 'ExperimenterResult']

log = logging.getLogger('assign_exp.exp')


class EffectivenessTerminator(MaximumFunctionCallTermination):
    """Terminator that terminates after a maximum number of function evaluations, and also calculates indicator values
    at every algorithm step."""

    def __init__(self, n_eval_max: int, metrics: List[Metric] = None):
        super(EffectivenessTerminator, self).__init__(n_max_evals=n_eval_max)

        self.metrics = metrics or []

    def update(self, algorithm):
        for metric in self.metrics:
            metric.calculate_step(algorithm)

        return super(EffectivenessTerminator, self).update(algorithm)


class ExperimenterResult(Result):
    """Result class with some extra data compared to the pymoo Result class."""

    def __init__(self):
        super(ExperimenterResult, self).__init__()

        self.plot_name = None
        self.metrics: Dict[str, Metric] = {}
        self.metric_converged = None
        self.termination: Optional[MetricTermination] = None

        self.n_steps = None
        self.n_steps_std = None

        self.n_eval = None
        self.n_eval_std = None

        self.exec_time_std = None

    @classmethod
    def from_result(cls, result: Result) -> 'ExperimenterResult':
        """Create from a Result class as output by an Algorithm."""
        exp_result = cls()
        for key in result.__dict__.keys():
            setattr(exp_result, key, getattr(result, key))

        exp_result.n_steps = len(exp_result.history) if exp_result.history is not None else None

        exp_result.n_eval = [algo.evaluator.n_eval for algo in exp_result.history]

        return exp_result

    @classmethod
    def aggregate_results(cls, results: List['ExperimenterResult'], align_end=False) -> 'ExperimenterResult':
        """Aggregate results from multiple ExperimenterResult instances, replacing metrics values with the mean and
        adding standard deviations."""
        result = cls()

        result.plot_name = results[0].plot_name
        result.exec_time, result.exec_time_std = cls._get_mean_std(results, lambda r: r.exec_time, align_end=align_end)
        result.n_steps, result.n_steps_std = cls._get_mean_std(results, lambda r: r.n_steps, align_end=align_end)
        result.n_eval, result.n_eval_std = cls._get_mean_std(results, lambda r: r.n_eval, align_end=align_end)

        for name, metric in results[0].metrics.items():
            result.metrics[name] = metric = copy.deepcopy(metric)

            metric.values, metric.values_std = {}, {}
            for key in metric.value_names:
                metric.values[key], metric.values_std[key] = \
                    cls._get_mean_std(results, lambda r: r.metrics[name].values[key], align_end=align_end)

        return result

    @staticmethod
    def _get_mean_std(results: List['ExperimenterResult'],
                      getter: Callable[['ExperimenterResult'], Optional[np.ndarray]], align_end=False) -> Tuple[np.ndarray, np.ndarray]:
        """Get mean and standard deviation for several repeated experimenter results."""

        results_data = None
        for result in results:
            res_data = getter(result)
            if res_data is None:
                continue
            res_data = np.atleast_3d(res_data)
            if results_data is None:
                results_data = res_data
            else:

                # Make sure results shapes are the same
                # Align data points at the end, so that mean values can be compared for steps from end
                # (e.g. upon termination)
                if results_data.shape[:2] != res_data.shape[:2]:
                    rs, r = results_data.shape, res_data.shape
                    new_shape = (max(rs[0], r[0]), max(rs[1], r[1]))

                    results_data_ = np.zeros(new_shape+(rs[2],))*np.nan
                    if align_end:
                        results_data_[-rs[0]:, -rs[1]:, :] = results_data
                    else:
                        results_data_[:rs[0], :rs[1], :] = results_data
                    results_data = results_data_

                    res_data_ = np.zeros(new_shape+(1,))*np.nan
                    if align_end:
                        res_data_[-r[0]:, -r[1]:, :] = res_data
                    else:
                        res_data_[:r[0], :r[1], :] = res_data
                    res_data = res_data_

                results_data = np.concatenate([results_data, res_data], axis=2)

        if results_data is None:
            return None, None

        mean_data = np.nanmean(results_data, axis=2)
        if mean_data.shape[0] == 1:
            mean_data = mean_data[0, :]
        # if len(mean_data) == 1:
        #     mean_data = mean_data[0]

        std_data = np.nanstd(results_data, axis=2)
        if std_data.shape[0] == 1:
            std_data = std_data[0, :]
        # if len(std_data) == 1:
        #     std_data = std_data[0]

        return mean_data, std_data

    @staticmethod
    def plot_compare_metrics(results: List['ExperimenterResult'], metric_name: str, plot_evaluations=False, **kwargs):
        metrics = [res.metrics[metric_name] for res in results]
        n_eval = [res.n_eval for res in results] if plot_evaluations else None
        if kwargs.get('titles') is None:
            kwargs['titles'] = [res.plot_name for res in results]
        Metric.plot_multiple(metrics, n_eval=n_eval, **kwargs)

    @staticmethod
    def plot_metrics_pareto(results: List['ExperimenterResult'], metric1_name_value: Tuple[str, str],
                            metric2_name_value: Tuple[str, str], names: List[str] = None, save_filename=None,
                            show=True):

        def _get_end_value_std(result: ExperimenterResult, metric_name: str, value_name: str) \
                -> Tuple[float, Optional[float]]:

            if metric_name == 'n_eval':
                value = result.n_eval[-1]
                std = result.n_eval_std[-1] if result.n_eval_std is not None else None
                return value, std

            metric = result.metrics[metric_name]
            value = metric.values[value_name][-1]
            std = metric.values_std[value_name][-1] if metric.values_std is not None else None
            return value, std

        plt.figure(), plt.title('%s.%s vs %s.%s' % (metric1_name_value[0], metric1_name_value[1], metric2_name_value[0],
                                                    metric2_name_value[1]))

        for i, res in enumerate(results):
            kwargs = {}
            if names is not None:
                kwargs['label'] = names[i]

            val1, std1 = _get_end_value_std(res, *metric1_name_value)
            val2, std2 = _get_end_value_std(res, *metric2_name_value)

            if std1 is not None and std2 is not None:
                plt.errorbar([val1], [val2], xerr=[std1], yerr=[std2], fmt='.', markersize=1, elinewidth=1, capsize=5,
                             **kwargs)
            else:
                plt.scatter([val1], [val2], s=1, marker='.', **kwargs)

        plt.xlabel('%s.%s' % (metric1_name_value[0], metric1_name_value[1]))
        plt.ylabel('%s.%s' % (metric2_name_value[0], metric2_name_value[1]))
        if names is not None:
            plt.legend(bbox_to_anchor=(1., 1.), loc='upper left')
        plt.tight_layout()

        if save_filename is not None:
            save_value_filename = save_filename+'_'+secure_filename(
                '_'.join([metric1_name_value[0], metric1_name_value[1], metric2_name_value[0],
                          metric2_name_value[1]]))
            plt.savefig(save_value_filename+'.png')
            plt.savefig(save_value_filename+'.svg')
        if show:
            plt.show()

    def plot_obj_progress(self, save_filename=None, show=True):
        plt.figure()
        plt.title(f'Objective progress: {self.plot_name}')

        for i, algo_step in enumerate(self.history):
            f_pareto = algo_step.opt.get('F')
            plt.scatter(f_pareto[:, 0], f_pareto[:, 1], s=3, label=f'Step {i}')
        plt.legend()

        if save_filename is not None:
            pl.savefig(save_filename+'.png')
            pl.savefig(save_filename+'.svg')
        if show:
            plt.show()

    def export_pandas(self) -> pd.DataFrame:
        has_std = self.n_eval_std is not None
        data = {
            'n_eval': self.n_eval,
        }
        if has_std:
            data['n_eval_std'] = self.n_eval_std

        for name, metric in self.metrics.items():
            data.update({'%s_%s' % (name, key): value for key, value in metric.results().items()})
            if has_std:
                data.update({'%s_%s_std' % (name, key): value for key, value in metric.results_std().items()})

        return pd.DataFrame(index=np.array(range(len(self.n_eval)), dtype=int)+1, data=data)

    def save_csv(self, path):
        res = self.export_pandas().to_csv(index_label='#').replace('\r', '')
        with open(path, 'w') as fp:
            fp.write(res)


class Experimenter:
    """Main class that handles the experiment for a given problem and algorithm."""

    results_folder: Optional[str] = None

    def __init__(self, problem: Problem, algorithm: Algorithm, n_eval_max: int, algorithm_name: str = None,
                 plot_name: str = None, metrics: List[Metric] = None, log_level='INFO', results_folder: str = None):
        self.problem = problem
        self.algorithm = algorithm
        self.algorithm_name = algorithm_name or algorithm.__class__.__name__
        self.plot_name = plot_name
        self.n_eval_max = n_eval_max

        if metrics is None:
            from assign_experiments.metrics import DeltaHVMetric, IGDMetric, SpreadMetric, MaxConstraintViolationMetric
            pf = problem.pareto_front()
            metrics = [
                DeltaHVMetric(pf),
                IGDMetric(pf),
                MaxConstraintViolationMetric(),
            ]
            # if problem.n_obj == 2:
            #     metrics.append(SpreadMetric())
        self.metrics = metrics

        self.results_folder = results_folder or self.results_folder  # Turn class attr into instance attr
        self._log_level = log_level

    # ## EFFECTIVENESS EXPERIMENTATION ## #

    def run_effectiveness_parallel(self, n_repeat: int):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.run_effectiveness, i) for i in range(n_repeat)]
            concurrent.futures.wait(futures)

            for fut in futures:
                fut.result()

    def run_effectiveness(self, repeat_idx: int = 0, seed=None) -> ExperimenterResult:
        """
        Run the effectiveness experiment: find out how well the algorithm is able to approach the Pareto front. Simply
        runs the algorithm with a predefines maximum number of function evaluations.
        """
        self.capture_log(level=self._log_level)

        termination = EffectivenessTerminator(n_eval_max=self.n_eval_max, metrics=self.metrics)

        # Run the algorithm
        log.info('Running effectiveness experiment:  %s / %s / %d' %
                 (self.problem.name(), self.algorithm_name, repeat_idx))
        result = minimize(
            self.problem, self.algorithm,
            termination=termination,
            copy_algorithm=True, copy_termination=True,
            seed=seed,
            save_history=True,
        )

        # Prepare experimenter results by including metrics
        result = ExperimenterResult.from_result(result)
        result.plot_name = self.plot_name or self.algorithm_name
        metrics: List[Metric] = result.algorithm.termination.metrics
        result.metrics = {met.name: met for met in metrics}

        # Reduce file size to prevent memory errors
        result.algorithm = None
        result.history = None

        # Store results and return
        result_path = self._get_effectiveness_result_path(repeat_idx=repeat_idx)
        with open(result_path, 'wb') as fp:
            fp.write(bz2.compress(pickle.dumps(result)))

        # log.info('Effectiveness experiment finished: %s / %s / %d' %
        #          (self.problem.name(), self.algorithm_name, repeat_idx))
        return result

    def get_effectiveness_result(self, repeat_idx: int) -> Optional[ExperimenterResult]:
        result_path = self._get_effectiveness_result_path(repeat_idx=repeat_idx)
        if not os.path.exists(result_path):
            return
        with open(result_path, 'rb') as fp:
            return pickle.loads(bz2.decompress(fp.read()))

    def get_effectiveness_results(self) -> List[ExperimenterResult]:
        results = []
        i = 0
        while True:
            result = self.get_effectiveness_result(repeat_idx=i)
            if result is None:
                break

            results.append(result)
            i += 1
        return results

    def has_aggregate_effectiveness_results(self) -> bool:
        return os.path.exists(self._get_agg_effectiveness_result_path())

    def get_aggregate_effectiveness_results(self, force=False, align_end=False) -> ExperimenterResult:
        """Returns results aggregated for all individual runs, using mean and std."""
        agg_results_path = self._get_agg_effectiveness_result_path()
        if not force and os.path.exists(agg_results_path):
            with open(agg_results_path, 'rb') as fp:
                return pickle.load(fp)

        log.info('Aggregating effectiveness results: %s / %s' % (self.problem.name(), self.algorithm_name))
        results = self.get_effectiveness_results()

        res = ExperimenterResult.aggregate_results(results, align_end=align_end)
        with open(agg_results_path, 'wb') as fp:
            pickle.dump(res, fp)
        return res

    def _get_effectiveness_result_path(self, repeat_idx: int) -> str:
        return self.get_problem_algo_results_path('result_%d.pkl' % repeat_idx)

    def _get_agg_effectiveness_result_path(self) -> str:
        return self.get_problem_algo_results_path('result_agg.pkl')

    # ## EFFICIENCY EXPERIMENTATION ## #

    def run_efficiency_repeated(self, metric_termination: MetricTermination) -> List[ExperimenterResult]:
        """Run efficiency experiments for the amount of previously generated effectiveness results available."""
        results = []

        i = 0
        while True:
            result = self.run_efficiency(metric_termination, repeat_idx=i)
            if result is None:
                break

            results.append(result)
            i += 1

        return results

    def run_efficiency(self, metric_termination: MetricTermination, repeat_idx: int) -> Optional[ExperimenterResult]:
        """
        Run the efficiency experiment: determine after how many steps an algorithm would terminate if a certain metric
        would have been used to detect convergence. Uses effectiveness results to "replay" an optimization session and
        returns results as if the passed metric would actually have been used.
        """
        self.capture_log(self._log_level)

        effectiveness_result = self.get_effectiveness_result(repeat_idx=repeat_idx)
        if effectiveness_result is None:
            return

        termination = copy.deepcopy(metric_termination)

        # Simulate algorithm execution using provided termination metric
        log.info('Running efficiency experiment: %s / %s / %s / %d' %
                 (self.problem.name(), self.algorithm_name, metric_termination.metric_name, repeat_idx))
        history = []
        result = None
        algorithm: Algorithm
        for algorithm in effectiveness_result.history:
            history.append(algorithm)

            if not termination.do_continue(algorithm):  # Metric convergence
                algorithm.history = history
                n_steps = len(history)

                result = algorithm.result()
                algorithm.termination = termination
                result.algorithm = algorithm

                # Modify metrics to reflect number of steps
                result = ExperimenterResult.from_result(result)
                result.plot_name = self.plot_name or self.algorithm_name
                result.metric_converged = True
                result.termination = termination

                result.metrics = metrics = {}
                for name, metric in effectiveness_result.metrics.items():
                    mod_metric = copy.deepcopy(metric)
                    mod_metric.values = {key: values[:n_steps] for key, values in mod_metric.values.items()}
                    metrics[name] = mod_metric
                break

        if result is None:  # Metric not converged
            result = copy.deepcopy(effectiveness_result)
            result.metric_converged = False
            result.termination = termination

        # Store results and return
        result_path = self._get_efficiency_result_path(metric_termination, repeat_idx=repeat_idx)
        with open(result_path, 'wb') as fp:
            fp.write(bz2.compress(pickle.dumps(result)))

        log.info('Efficiency experiment finished (converged: %r): %s / %s / %d' %
                 (result.metric_converged, self.problem.name(), self.algorithm_name, repeat_idx))
        return result

    def get_efficiency_result(self, metric_termination: MetricTermination, repeat_idx: int) \
            -> Optional[ExperimenterResult]:
        result_path = self._get_efficiency_result_path(metric_termination, repeat_idx=repeat_idx)
        if not os.path.exists(result_path):
            return
        with open(result_path, 'rb') as fp:
            return pickle.loads(bz2.decompress(fp.read()))

    def get_aggregate_efficiency_results(self, metric_termination: MetricTermination, force=False) \
            -> ExperimenterResult:
        """Get efficiency results aggregated for all efficiency experiment runs."""
        agg_results_path = self._get_agg_efficiency_result_path(metric_termination)
        if not force and os.path.exists(agg_results_path):
            with open(agg_results_path, 'rb') as fp:
                return pickle.load(fp)

        log.info('Aggregating efficiency results: %s / %s / %s' %
                 (self.problem.name(), self.algorithm_name, metric_termination.metric_name))
        results = self.get_list_efficiency_results(metric_termination)

        res = ExperimenterResult.aggregate_results(results)
        with open(agg_results_path, 'wb') as fp:
            pickle.dump(res, fp)
        return res

    def get_list_efficiency_results(self, metric_termination: MetricTermination) -> List[ExperimenterResult]:
        """Get efficiency results aggregated for all efficiency experiment runs."""
        results = []
        i = 0
        while True:
            result = self.get_efficiency_result(metric_termination, repeat_idx=i)
            if result is None:
                break

            results.append(result)
            i += 1
        return results

    def _get_efficiency_result_path(self, metric_termination: MetricTermination, repeat_idx: int) -> str:
        return self.get_problem_algo_metric_results_path(metric_termination, 'result_%d.pkl' % (repeat_idx,))

    def _get_agg_efficiency_result_path(self, metric_termination: MetricTermination) -> str:
        return self.get_problem_algo_metric_results_path(metric_termination, 'result_agg.pkl')

    # ## HELPER FUNCTIONS ## #

    @staticmethod
    @contextlib.contextmanager
    def temp_results():
        """
        Sets a temporary folder as results folder. Useful for running experiments and directly analyzing the
        results. Usage:

        with Experimenter.temp_results():
            ...
        """

        orig_res_folder = Experimenter.results_folder

        with tempfile.TemporaryDirectory() as tmp_dir:
            Experimenter.results_folder = tmp_dir
            yield

        Experimenter.results_folder = orig_res_folder

    def get_problem_algo_results_path(self, sub_path: str = None) -> str:
        problem_algo_path = '%s/%s' % (secure_filename(self.problem.name()), secure_filename(self.algorithm_name))
        if sub_path is not None:
            problem_algo_path += '/'+sub_path
        return self._get_results_path(problem_algo_path)

    def get_problem_algo_metric_results_path(self, metric_termination: MetricTermination, sub_path: str = None) -> str:
        path = secure_filename(metric_termination.metric_name)
        if sub_path is not None:
            path = os.path.join(path, sub_path)
        return self.get_problem_algo_results_path(path)

    def _get_results_path(self, sub_path: str = None) -> str:
        if self.results_folder is None:
            raise ValueError('Must set results_folder on the class!')

        path = self.results_folder
        if sub_path is not None:
            path = os.path.join(path, sub_path)

        os.makedirs(os.path.dirname(path) if sub_path is not None else path, exist_ok=True)
        return path

    @staticmethod
    def capture_log(level='INFO'):
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': True,
            'formatters': {
                'console': {
                    'format': '%(levelname)- 8s %(asctime)s %(name)- 18s: %(message)s'
                },
            },
            'handlers': {
                'console': {
                    'level': level,
                    'class': 'logging.StreamHandler',
                    'formatter': 'console',
                },
            },
            'loggers': {
                'assign_enc': {
                    'handlers': ['console'],
                    'level': level,
                },
                'assign_exp': {
                    'handlers': ['console'],
                    'level': level,
                },
            },
        })
