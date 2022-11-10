import numpy as np
from typing import *
import matplotlib.pyplot as plt
from assign_enc.encoding import *
from assign_pymoo.problem import *

__all__ = ['MetricsComparer']


class MetricsComparer:

    def __init__(self, n_samples: int = None, n_leave_out: int = None):
        self.n_samples = n_samples or 20
        self.n_leave_out = n_leave_out or int(self.n_samples*.10)

    def compare_encoders(self, problem: AssignmentProblem, encoders: List[Encoder], plot=True, show=True):
        points = np.empty((len(encoders), 4))
        labels = []
        for i, encoder in enumerate(encoders):
            print(f'Evaluating {encoder!s} ({i+1}/{len(encoders)})')
            points[i, :] = self.get_metrics(problem, encoder)
            print(f'           imp ratio = {points[i, 0]:.2f}; inf err = {points[i, 2]:.2f}')
            labels.append(str(encoder))

        if plot:
            title = f'Encoder Comparison for Problem:\n{problem}'
            self.plot(points, labels, title, show=show)
        return points, labels

    def compare_problems(self, problems: List[AssignmentProblem], encoder: Encoder, plot=True, show=True):
        points = np.empty((len(problems), 4))
        labels = []
        for i, problem in enumerate(problems):
            print(f'Evaluating {problem!s} ({i+1}/{len(problems)})')
            points[i, :] = self.get_metrics(problem, encoder)
            labels.append(str(problem))

        if plot:
            title = f'Problem Comparison for Encoder:\n{encoder}'
            self.plot(points, labels, title, show=show)
        return points, labels

    @classmethod
    def plot(cls, points: np.ndarray, labels: List[str], title: str, show=True):
        """Points is n x 4 matrix; where columns are: imp ratio, imp ratio std dev, inf error, inf error std dev"""
        plt.figure()
        plt.title(title)

        # x, x_err, y, y_err = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        # plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='.k', capsize=1)
        x, y = points[:, 0], points[:, 2]
        plt.scatter(x, y, c='k', marker='.')
        is_log = False
        if np.max(x) > 10:
            plt.gca().set_xscale('log')
            is_log = True
        for i, label in enumerate(labels):
            plt.text(points[i, 0], points[i, 2], label)
        plt.xlabel(f'Imputation Ratio{" (log)" if is_log else ""}')
        plt.ylabel('Relative information error')

        if show:
            plt.show()

    def get_metrics(self, problem: AssignmentProblem, encoder: Encoder) -> np.ndarray:
        """Calculates imputation ratio and information error"""
        problem = self._get_for_encoder(problem, encoder)

        metrics = np.zeros((4,))
        metrics[0] = self.get_imputation_ratio(problem)
        metrics[2:] = self.get_information_error(problem)

        return metrics

    @staticmethod
    def get_imputation_ratio(problem: AssignmentProblem):
        return problem.get_imputation_ratio()

    def get_information_error(self, problem: AssignmentProblem, n_samples: int = None, n_leave_out: int = None,
                              **kwargs) -> np.ndarray:
        """Get the max information error (and its std dev) of each problem output"""
        if n_samples is None:
            n_samples = self.n_samples
        if n_leave_out is None:
            n_leave_out = self.n_leave_out
        information_errors = problem.get_information_error(n_samples=n_samples, n_leave_out=n_leave_out, **kwargs)
        try:
            i_max = np.argmax(information_errors[0, :])
            return information_errors[:, i_max]
        except ValueError:
            return np.ones((2,))

    @staticmethod
    def _get_for_encoder(problem: AssignmentProblem, encoder: Encoder) -> AssignmentProblem:
        return problem.get_for_encoder(encoder)
