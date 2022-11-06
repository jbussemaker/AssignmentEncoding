import copy
import numpy as np
from assign_pymoo.problem import *
from smt.surrogate_models.krg import KRG
from pymoo.core.evaluator import Evaluator
from smt.surrogate_models.surrogate_model import SurrogateModel

__all__ = ['InformationContentAnalyzer']


class InformationContentAnalyzer:
    """
    Some helper functions for determining the level of information content that can be represented using some design
    variable encoding scheme, measuring by how good some surrogate model is at predicting output from input. Output is
    given by some test problem.

    Surrogate model performance is measuring using LOOCV (Leave-One-Out-Cross-Validation): the RMSE of errors for
    leaving one point out and retraining at a time.
    """

    def __init__(self, problem: AssignmentProblem, surrogate_model: SurrogateModel = None):
        self._problem = problem
        if surrogate_model is None:
            surrogate_model = self._get_default_surrogate_model()
        self._surrogate_model = surrogate_model

    @staticmethod
    def _get_default_surrogate_model():
        model = KRG(theta0=[1e-6], print_global=False)
        return model

    def get_information_error(self, n_samples: int, n_leave_out: int = None) -> np.ndarray:
        """Calculate information error (LOOCV metric) for each problem output (f and g).
        Returns 2 x n_out array, where first row is mean, second row is std dev."""

        # Get and evaluate sampling points
        init_sampler = self._problem.get_init_sampler()
        pop = init_sampler.do(self._problem, n_samples)
        pop = Evaluator().eval(self._problem, pop)

        x_train = pop.get('X')
        y_train = pop.get('F')
        if pop.get('G') is not None:
            y_train = np.column_stack([y_train, pop.get('G')])
            not_all_zero = ~np.all(y_train == 0, axis=0)
            y_train = y_train[:, not_all_zero]

        if x_train.shape[0] <= 1:
            return np.ones((2, y_train.shape[1]))

        # Calculate LOOCV
        return self.cross_validate(x_train, y_train, n_leave_out=n_leave_out)

    def cross_validate(self, x_train: np.ndarray, y_train: np.ndarray, n_leave_out: int = None) -> np.ndarray:
        """Cross-validate by leaving n_leave_out (if not given: n_samples) out and retraining;
        returns vector of size n_y with relative RMSE for each output."""
        if n_leave_out is None:
            n_leave_out = x_train.shape[0]
        if n_leave_out > x_train.shape[0]:
            n_leave_out = x_train.shape[0]

        # Loop over points left out
        i_leave_out = np.random.choice(x_train.shape[0], n_leave_out, replace=False)
        errors = np.empty((n_leave_out, y_train.shape[1]))
        for i, i_pt in enumerate(i_leave_out):
            errors[i, :] = self._get_error(x_train, y_train, i_pt)

        # Get RMSE over all errors
        rmse = np.sqrt(np.mean(errors**2, axis=0))
        rmse_std_dev = np.sqrt(np.std(errors**2, axis=0))
        return np.array([rmse, rmse_std_dev])

    def _get_error(self, x_train: np.ndarray, y_train: np.ndarray, i_leave_out: int) -> np.ndarray:
        # Separate samples
        x_leave_out = x_train[i_leave_out, :]
        y_leave_out = y_train[i_leave_out, :]
        x_train_lo = np.delete(x_train, i_leave_out, axis=0)
        y_train_lo = np.delete(y_train, i_leave_out, axis=0)

        # Retrain model
        surrogate_model = self._copy_surrogate_model(self._surrogate_model)
        surrogate_model.set_training_values(x_train_lo, y_train_lo)
        surrogate_model.train()

        # Calculate prediction error
        y_range = np.max(y_train, axis=0)-np.min(y_train, axis=0)
        y_lo_predict = surrogate_model.predict_values(np.atleast_2d(x_leave_out))[0, :]
        y_lo_error = (y_lo_predict-y_leave_out)/y_range
        return y_lo_error

    @staticmethod
    def _copy_surrogate_model(surrogate_model: SurrogateModel) -> SurrogateModel:
        return copy.deepcopy(surrogate_model)
