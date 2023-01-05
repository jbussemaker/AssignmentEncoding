import timeit
import logging
import numpy as np
import pandas as pd
from assign_pymoo.algo import *
from assign_pymoo.problem import *
from assign_pymoo.sampling import *
from assign_experiments.runner import *
from assign_enc.encoder_registry import *
from assign_experiments.experimenter import *
from assign_experiments.problems.analytical import *

log = logging.getLogger('assign_exp.exp01')


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


def run_experiment(n_repeat=8, n_sample=1000, do_run=True):
    Experimenter.capture_log()
    pop_size = 50
    n_gen = 6

    problems, algorithms, plot_names = [], [], []
    for eager in [True, False]:
        for very_high in [False, True]:
            if eager:
                for i, base_eager_problem in enumerate([
                    get_very_high_imputation_ratio_problem(),
                ] if very_high else [get_high_imputation_ratio_problem()]):
                    prob_name = 'H Eager' if not very_high else 'VH Eager'
                    for eager_imputer in EAGER_IMPUTERS:
                        imputer = eager_imputer()
                        problem = base_eager_problem.get_for_encoder(
                            base_eager_problem.assignment_manager.encoder.get_for_imputer(imputer))
                        problems.append(problem)
                        algorithms.append(get_ga_algo(problem, pop_size=pop_size))

                        plot_names.append(f'{prob_name}: {imputer!s}')
            else:
                for i, base_lazy_problem in enumerate([
                    get_very_high_imputation_ratio_lazy_problem(),
                ] if very_high else [get_high_imputation_ratio_lazy_problem()]):
                    prob_name = 'H Lazy' if not very_high else 'VH Lazy'
                    for lazy_imputer in LAZY_IMPUTERS:
                        imputer = lazy_imputer()
                        problem = base_lazy_problem.get_for_encoder(
                            base_lazy_problem.assignment_manager.encoder.get_for_imputer(imputer))
                        problems.append(problem)
                        algorithms.append(get_ga_algo(problem, pop_size=pop_size))

                        plot_names.append(f'{prob_name}: {imputer!s}')

    stats = {'prob': [], 'enc': [], 'imp': [], 'n': [], 'n_des_space': [], 'imp_ratio': [], 'n_dv': [], 'inf_idx': [],
             'sample_time_sec': [], 'sample_time_sec_std': [], 'frac_imputed': [], 'frac_imputed_std': [],
             'hv_doe': [], 'hv_doe_std': [], 'hv_end': [], 'hv_end_std': []}
    for i, enc_prob in enumerate(problems):
        log.info(f'Timing {i+1}/{len(problems)} {enc_prob!s} {enc_prob.assignment_manager.encoder!s}')
        sampling_times, frac_imputed = _time_imputer(enc_prob, n_repeat, n_sample)
        stats['sample_time_sec'].append(np.mean(sampling_times))
        stats['sample_time_sec_std'].append(np.std(sampling_times))
        stats['frac_imputed'].append(np.mean(frac_imputed))
        stats['frac_imputed_std'].append(np.std(frac_imputed))

        encoder = enc_prob.assignment_manager.encoder
        stats['prob'].append(str(enc_prob))
        stats['enc'].append(str(encoder))
        stats['imp'].append(str(encoder._imputer))
        stats['n'].append(enc_prob.get_n_valid_design_points())
        stats['n_des_space'].append(enc_prob.get_n_design_points())
        stats['imp_ratio'].append(enc_prob.get_imputation_ratio())
        stats['n_dv'].append(len(encoder.design_vars))
        stats['inf_idx'].append(enc_prob.get_information_index())
        stats['hv_doe'].append(np.nan)
        stats['hv_doe_std'].append(np.nan)
        stats['hv_end'].append(np.nan)
        stats['hv_end_std'].append(np.nan)

    n_eval = pop_size*n_gen
    algo_names = ['NSGA2']*len(algorithms)

    exp_name = '01_imputation'
    set_results_folder(exp_name)
    res_folder = Experimenter.results_folder
    df = pd.DataFrame(data=stats)
    df.to_csv(f'{res_folder}/stats_init.csv')

    if do_run:
        run(exp_name, problems, algorithms, algo_names=algo_names, plot_names=plot_names, n_repeat=n_repeat,
            n_eval_max=n_eval, do_run=do_run)

    exp = run(exp_name, problems, algorithms, algo_names=algo_names, plot_names=plot_names, n_repeat=n_repeat,
              n_eval_max=n_eval, return_exp=True)
    for i, experimenter in enumerate(exp):
        i_stats = i
        try:
            agg_res = experimenter.get_aggregate_effectiveness_results()
        except IndexError:
            log.info(f'Results not available for: {experimenter.problem.name()} / {experimenter.algorithm_name}')
            continue
        df.at[i_stats, 'hv_doe'] = agg_res.metrics['delta_hv'].values['delta_hv'][0]
        df.at[i_stats, 'hv_doe_std'] = agg_res.metrics['delta_hv'].values_std['delta_hv'][0]
        df.at[i_stats, 'hv_end'] = agg_res.metrics['delta_hv'].values['delta_hv'][-1]
        df.at[i_stats, 'hv_end_std'] = agg_res.metrics['delta_hv'].values_std['delta_hv'][-1]
    df['hv_ratio'] = df['hv_end']/df['hv_doe']
    df['hv_ratio_std'] = df['hv_end_std']/df['hv_doe']
    stats_file = f'{res_folder}/stats.csv'
    df.to_csv(stats_file)


def _time_imputer(problem: AssignmentProblem, n_repeat: int, n_sample: int):
    sampling_times = []
    frac_imputed = []
    for _ in range(n_repeat):
        problem = problem.get_for_encoder(problem.assignment_manager.encoder)
        sampling = RepairedRandomSampling(repair=problem.get_repair())
        sampling.track_x_last_init = True

        s = timeit.default_timer()
        x_sampled = sampling.do(problem, n_sample).get('X')
        sampling_times.append(timeit.default_timer()-s)

        n_total = x_sampled.shape[0]
        x_init = sampling.x_last_init
        n_imputed = np.sum(np.any(x_init != x_sampled, axis=1))
        frac_imputed.append(n_imputed/n_total)

    return sampling_times, frac_imputed


if __name__ == '__main__':
    # Check problem sizes to tune them for approximately the same design space sizes: problems with similar design space
    # sizes should also be similar in the impact of the population size and nr of generations
    # show_problem_sizes(), exit()

    run_experiment(n_repeat=16)
