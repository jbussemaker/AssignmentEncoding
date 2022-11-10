from assign_pymoo.algo import *
from assign_experiments.runner import *
from assign_experiments.encoders import *
from assign_experiments.problems.analytical import *


def get_high_imputation_ratio_problem():
    encoder = AmountFirstGroupedEncoder(DEFAULT_EAGER_IMPUTER(), SourceAmountGrouper(), OneVarLocationGrouper())
    return AnalyticalPartitioningProblem(encoder, n_src=4, n_tgt=5)


def get_high_imputation_ratio_lazy_problem():
    encoder = LazyAmountFirstEncoder(DEFAULT_LAZY_IMPUTER(), SourceLazyAmountEncoder(), FlatLazyConnectionEncoder())
    return AnalyticalPartitioningProblem(encoder, n_src=4, n_tgt=5)


def get_very_high_imputation_ratio_problem():
    encoder = AmountFirstGroupedEncoder(DEFAULT_EAGER_IMPUTER(), TotalAmountGrouper(), CoordIndexLocationGrouper())
    return AnalyticalPartitioningProblem(encoder, n_src=4, n_tgt=5)


def get_very_high_imputation_ratio_lazy_problem():
    encoder = LazyDirectMatrixEncoder(DEFAULT_LAZY_IMPUTER())
    return AnalyticalPartitioningProblem(encoder, n_src=4, n_tgt=3)


def show_problem_sizes():
    show_problem_size(get_high_imputation_ratio_problem())
    show_problem_size(get_high_imputation_ratio_lazy_problem())
    show_problem_size(get_very_high_imputation_ratio_problem())
    show_problem_size(get_very_high_imputation_ratio_lazy_problem())


def run_experiment(very_high=False, eager=True, n_repeat=8, do_run=True):
    pop_size = 50
    n_gen = 6

    problems, algorithms, plot_names = [], [], []
    if eager:
        for i, base_eager_problem in enumerate([
            # get_high_imputation_ratio_problem(),
            get_very_high_imputation_ratio_problem(),
        ] if very_high else [get_high_imputation_ratio_problem()]):
            prob_name = 'H Eager' if i == 0 else 'VH Eager'
            for eager_imputer in EAGER_IMPUTERS:
                imputer = eager_imputer()
                problem = base_eager_problem.get_for_encoder(
                    base_eager_problem.assignment_manager.encoder.get_for_imputer(imputer))
                problems.append(problem)
                algorithms.append(get_ga_algo(problem, pop_size=pop_size))

                plot_names.append(f'{prob_name}: {imputer!s}')

    else:
        for i, base_lazy_problem in enumerate([
            # get_high_imputation_ratio_lazy_problem(),
            get_very_high_imputation_ratio_lazy_problem(),
        ] if very_high else [get_high_imputation_ratio_lazy_problem()]):
            prob_name = 'H Lazy' if i == 0 else 'VH Lazy'
            for lazy_imputer in LAZY_IMPUTERS:
                imputer = lazy_imputer()
                problem = base_lazy_problem.get_for_encoder(
                    base_lazy_problem.assignment_manager.encoder.get_for_imputer(imputer))
                problems.append(problem)
                algorithms.append(get_ga_algo(problem, pop_size=pop_size))

                plot_names.append(f'{prob_name}: {imputer!s}')

    n_eval = pop_size*n_gen
    algo_names = ['NSGA2']*len(algorithms)
    name = 'eager' if eager else 'lazy'
    if very_high:
        name += '_vh'
    return run(f'01_imputation_{name}', problems, algorithms, algo_names=algo_names, plot_names=plot_names,
               n_repeat=n_repeat, n_eval_max=n_eval, do_run=do_run)


if __name__ == '__main__':
    # Check problem sizes to tune them for approximately the same design space sizes: problems with similar design space
    # sizes should also be similar in the impact of the population size and nr of generations
    show_problem_sizes(), exit()

    # run_experiment(eager=True, very_high=True, n_repeat=8)
    # run_experiment(eager=True, very_high=False, n_repeat=8)
    run_experiment(eager=False, very_high=True, n_repeat=8)
    run_experiment(eager=False, very_high=False, n_repeat=8)
