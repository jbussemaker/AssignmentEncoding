from assign_pymoo.algo import *
from assign_experiments.runner import *
from assign_experiments.encoders import *
from assign_pymoo.metrics_compare import *
from assign_experiments.problems.analytical import *


def get_vary_imputation_ratio_problem():
    problem = AnalyticalConnectingProblem(DEFAULT_EAGER_ENCODER(), n=4)
    encoders = [
        AmountFirstGroupedEncoder(DEFAULT_EAGER_IMPUTER(), SourceAmountGrouper(), OneVarLocationGrouper()),
        LazyAmountFirstEncoder(DEFAULT_LAZY_IMPUTER(), SourceTargetLazyAmountEncoder(), FlatLazyConnectionEncoder()),
    ]
    return problem, encoders


def get_vary_information_error_problem():
    problem = AnalyticalConnectingProblem(DEFAULT_EAGER_ENCODER(), n=4)
    encoders = [
        DirectMatrixEncoder(DEFAULT_EAGER_IMPUTER()),
        OneVarEncoder(DEFAULT_EAGER_IMPUTER()),
    ]
    return problem, encoders


def get_vary_information_index_problem():
    problem = AnalyticalConnectingProblem(DEFAULT_EAGER_ENCODER(), n=4)
    encoders = [
        OneVarEncoder(DEFAULT_EAGER_IMPUTER()),
        DirectMatrixEncoder(DEFAULT_EAGER_IMPUTER()),
    ]
    return problem, encoders


def show_problem_sizes():
    show_problem_size(get_vary_imputation_ratio_problem()[0])
    show_problem_size(get_vary_information_error_problem()[0])
    show_problem_size(get_vary_information_index_problem()[0])


def verify_metric_trends(imp_ratio=False, inf_idx=False):
    mode = 'Eager'
    metric = 'Imp Ratio' if imp_ratio else 'Inf Error'
    print(f'{mode} encoding; mainly varying metric: {metric}')
    if imp_ratio:
        problem, encoders = get_vary_imputation_ratio_problem()
    elif inf_idx:
        problem, encoders = get_vary_information_index_problem()
    else:
        problem, encoders = get_vary_information_error_problem()
    MetricsComparer(n_samples=50, n_leave_out=30).compare_encoders(problem, encoders, inf_idx=inf_idx or imp_ratio)


def run_experiment(imp_ratio=False, inf_idx=False, sbo=False, n_repeat=8, do_run=True):
    pop_size = 50
    n_gen = 15
    n_infill = 30

    problems, algorithms, plot_names = [], [], []
    if imp_ratio:
        prob_name = 'Imp Ratio'
        base_problem, encoders = get_vary_imputation_ratio_problem()
    elif inf_idx:
        prob_name = 'Inf Idx'
        base_problem, encoders = get_vary_information_index_problem()
    else:
        prob_name = 'Inf Error'
        base_problem, encoders = get_vary_information_error_problem()
    for encoder in encoders:
        problem = base_problem.get_for_encoder(encoder)
        problems.append(problem)
        plot_names.append(f'{prob_name}: {encoder!s}')

        if sbo:
            algorithms.append(get_sbo_algo(problem, init_size=pop_size))
        else:
            algorithms.append(get_ga_algo(problem, pop_size=pop_size))

    n_eval = pop_size+n_infill if sbo else pop_size*n_gen
    algo_names = ['SBO' if sbo else 'NSGA2']*len(algorithms)
    name = 'eager'
    prob_name = prob_name.lower().replace(' ', '_')
    return run(f'02_metric_corr_{prob_name}_{algo_names[0].lower()}_{name}', problems, algorithms,
               algo_names=algo_names, plot_names=plot_names, n_repeat=n_repeat, n_eval_max=n_eval, do_run=do_run)


if __name__ == '__main__':
    # Check problem sizes to tune them for approximately the same design space sizes: problems with similar design space
    # sizes should also be similar in the impact of the population size and nr of generations
    # show_problem_sizes(), exit()
    # verify_metric_trends(imp_ratio=True), exit()
    # verify_metric_trends(imp_ratio=False), exit()
    # verify_metric_trends(inf_idx=True), exit()

    run_experiment(imp_ratio=True, sbo=False, n_repeat=8)
    run_experiment(imp_ratio=False, sbo=False, n_repeat=8)
    run_experiment(inf_idx=True, sbo=False, n_repeat=8)
    run_experiment(imp_ratio=True, sbo=True, n_repeat=8)
    run_experiment(imp_ratio=False, sbo=True, n_repeat=8)
    run_experiment(inf_idx=True, sbo=True, n_repeat=8)
