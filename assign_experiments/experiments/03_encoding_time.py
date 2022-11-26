import os
import timeit
import numpy as np
import pandas as pd
from assign_pymoo.problem import *
from assign_pymoo.sampling import *
from assign_experiments.runner import *
from assign_enc.encoder_registry import *
from werkzeug.utils import secure_filename
from assign_experiments.experimenter import *
from assign_experiments.problems.analytical import *
import matplotlib.pyplot as plt

N_SAMPLES = [10, 100, 1000]


def get_problem_lazy_val():  # Main time influence by time it takes to validate lazy-decoded matrices
    problems = [
        AnalyticalConnectingProblem(DEFAULT_EAGER_ENCODER(), n=4),
        AnalyticalConnectingProblem(DEFAULT_EAGER_ENCODER(), n=3),
        AnalyticalConnectingProblem(DEFAULT_EAGER_ENCODER(), n=2),
    ]
    eager_encoder = DirectMatrixEncoder(DEFAULT_EAGER_IMPUTER())
    lazy_encoder = LazyDirectMatrixEncoder(DEFAULT_LAZY_IMPUTER())
    return problems, eager_encoder, lazy_encoder


def get_problem_lazy_get_matrix():  # Main time influence by generating matrices on demand for lazy encoder
    problems = [
        AnalyticalPermutingProblem(DEFAULT_EAGER_ENCODER(), n=7),
        AnalyticalPermutingProblem(DEFAULT_EAGER_ENCODER(), n=6),
        AnalyticalPermutingProblem(DEFAULT_EAGER_ENCODER(), n=5),
        AnalyticalPermutingProblem(DEFAULT_EAGER_ENCODER(), n=4),
    ]
    eager_encoder = AmountFirstGroupedEncoder(DEFAULT_EAGER_IMPUTER(), TotalAmountGrouper(), OneVarLocationGrouper())
    lazy_encoder = LazyAmountFirstEncoder(DEFAULT_LAZY_IMPUTER(), FlatLazyAmountEncoder(), FlatLazyConnectionEncoder())
    return problems, eager_encoder, lazy_encoder


def show_problem_sizes():
    for problems, eager_encoder, lazy_encoder in [get_problem_lazy_val(), get_problem_lazy_get_matrix()]:
        for i, problem in enumerate(problems):
            print(f'Problem variant {i}')
            show_problem_size(problem)
            print(f'Eager imputation ratio: {problem.get_for_encoder(eager_encoder).get_imputation_ratio()}')
            print(f'Lazy  imputation ratio: {problem.get_for_encoder(lazy_encoder).get_imputation_ratio()}')
            print('')


def run_experiment(n_repeat=10):
    set_results_folder('03_encoding_time')
    results_folder = Experimenter.results_folder

    run_n_sample_dep_exp(results_folder, n_repeat=n_repeat)
    run_n_matrix_dep_exp(results_folder, n_repeat=n_repeat)


def run_n_sample_dep_exp(results_folder, n_repeat=10):
    n_samples = N_SAMPLES
    x_values = []
    encoding_res = []
    sampling_res = []
    y_labels = []
    for (problems, eager_encoder, lazy_encoder), problem_name in [
        (get_problem_lazy_val(), 'Validation Influence'),
        (get_problem_lazy_get_matrix(), 'Matrix Gen Influence'),
    ]:
        problem = problems[0]
        for encoder, enc_name in [(eager_encoder, 'Eager'), (lazy_encoder, 'Lazy')]:
            encoding_time_series = np.zeros((len(n_samples), 2))
            sampling_time_series = np.zeros((len(n_samples), 2))
            for i, n in enumerate(n_samples):
                print(f'Timing {problem_name} {enc_name} @ {n} samples')
                times = _time_encoder(problem, encoder, n_repeat, n)
                encoding_time_series[i, :] = times[:2]
                sampling_time_series[i, :] = times[2:]
            x_values.append(n_samples)
            encoding_res.append(encoding_time_series)
            sampling_res.append(sampling_time_series)
            y_labels.append(f'{enc_name}; {problem_name}')

    plot_timing_results(results_folder, 'Samples vs Encoding Time', x_values, 'Samples', encoding_res, y_labels)
    plot_timing_results(results_folder, 'Samples vs Sampling Time', x_values, 'Samples', sampling_res, y_labels)


def run_n_matrix_dep_exp(results_folder, n_repeat=10):
    n_samples = N_SAMPLES[-1]
    x_values = []
    encoding_res = []
    sampling_res = []
    y_labels = []
    for (problems, eager_encoder, lazy_encoder), problem_name in [
        (get_problem_lazy_val(), 'Validation Influence'),
        (get_problem_lazy_get_matrix(), 'Matrix Gen Influence'),
    ]:
        n_mat = [problem.assignment_manager.matrix_gen.count_all_matrices() for problem in problems]
        for encoder, enc_name in [(eager_encoder, 'Eager'), (lazy_encoder, 'Lazy')]:
            encoding_time_series = np.zeros((len(problems), 2))
            sampling_time_series = np.zeros((len(problems), 2))
            for i, problem in enumerate(problems):
                print(f'Timing {enc_name} problem {i} @ {n_mat[i]} matrices')
                times = _time_encoder(problem, encoder, n_repeat, n_samples)
                encoding_time_series[i, :] = times[:2]
                sampling_time_series[i, :] = times[2:]
            x_values.append(n_mat)
            encoding_res.append(encoding_time_series)
            sampling_res.append(sampling_time_series)
            y_labels.append(f'{enc_name}; {problem_name}')

    plot_timing_results(results_folder, 'Nr Matrices vs Encoding Time', x_values, 'Nr of matrices', encoding_res, y_labels)
    plot_timing_results(results_folder, 'Nr Matrices vs Sampling Time', x_values, 'Nr of matrices', sampling_res, y_labels)


def plot_timing_results(results_folder, title, x_values, x_label, y_values, y_labels):
    for div_by_x in [False, True]:
        for plot_log in [True, False]:
            filename = os.path.join(results_folder, secure_filename(title.lower()))
            if div_by_x:
                filename += '_by_x'
            if not plot_log:
                filename += '_lin'
            fig = plt.figure(figsize=(12, 6))
            plt.title(f'{title} per {x_label}' if div_by_x else title)

            y_norm = np.min(y_values[0][:, 0])
            plotted_data = []
            col_names = []
            for i, y_ser in enumerate(y_values):
                x = x_values[i]
                y = y_ser[:, 0]/y_norm
                y_err = y_ser[:, 1]/y_norm
                if div_by_x:
                    y /= x
                    y_err /= x
                plt.errorbar(x, y, yerr=y_err, fmt='--.', capsize=3, elinewidth=.5,
                             label=y_labels[i])
                plotted_data += [x, y, y_err]
                col_names += [f'{y_labels[i]} ({x_label})', y_labels[i], f'{y_labels[i]} (Std Dev)']
            if plot_log:
                plt.gca().set_xscale('log')
                plt.gca().set_yscale('log')
            plt.xlabel(x_label), plt.ylabel('Relative time')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()

            if plot_log and not div_by_x:
                n_max = max([len(x) for x in x_values])
                data = np.zeros((n_max, len(col_names)))*np.nan
                for i, values in enumerate(plotted_data):
                    data[:len(values), i] = values
                data = pd.DataFrame(data=data, columns=col_names)
                data.to_csv(filename+'.csv')

            plt.savefig(filename+'.png')
            plt.savefig(filename+'.svg')
            plt.close(fig)


def _time_encoder(problem: AssignmentProblem, encoder, n_repeat: int, n_sample: int, report=False):
    encoding_times = []
    sampling_times = []
    for _ in range(n_repeat):
        s = timeit.default_timer()
        problem = problem.get_for_encoder(encoder)
        encoding_times.append(timeit.default_timer()-s)

        s = timeit.default_timer()
        sampling = RepairedRandomSampling(repair=problem.get_repair())
        sampling.do(problem, n_sample)
        sampling_times.append(timeit.default_timer()-s)

    encoding_mean, encoding_std = np.mean(encoding_times), np.std(encoding_times)
    sampling_mean, sampling_std = np.mean(sampling_times), np.std(sampling_times)
    if report:
        print(f'Encoder (x{n_repeat}): {encoder!s}')
        print(f'Encoding time: {np.mean(encoding_times):.2g} +- {np.std(encoding_times):.2g} sec')
        print(f'Sampling time: {np.mean(sampling_times):.2g} +- {np.std(sampling_times):.2g} sec ({n_sample} samples)')
    return np.array([encoding_mean, encoding_std, sampling_mean, sampling_std])


if __name__ == '__main__':
    # show_problem_sizes(), exit()
    run_experiment(n_repeat=5)
