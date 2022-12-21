import os
import enum
import timeit
import logging
import numpy as np
import pandas as pd
from typing import *
from assign_pymoo.algo import *
from assign_enc.time_limiter import *
from assign_experiments.runner import *
from assign_enc.encoding import Encoder
from assign_enc.encoder_registry import *
from assign_pymoo.problem import AssignmentProblem
from assign_experiments.problems.analytical import *
from assign_experiments.experimenter import Experimenter
import matplotlib.pyplot as plt

log = logging.getLogger('assign_exp.exp04')


class Size(enum.Enum):
    SM = 1  # Target: ~100 design points
    MD = 2  # ~5 000 design points
    LG = 3  # ~100 000 design points


def _get_problem_factory(cls, **kwargs) -> Callable[[Encoder], AssignmentProblem]:
    def _problem_factory(enc):
        return cls(enc, **kwargs)
    return _problem_factory


def get_problems(size: Size, return_factories=False) -> Union[List[AssignmentProblem], List[Callable[[Encoder], AssignmentProblem]]]:
    init_enc = lambda: OneVarEncoder(DEFAULT_EAGER_IMPUTER())
    prob_kwargs = [
        (AnalyticalCombinationProblem, {Size.SM: {'n_tgt': 100}, Size.MD: {'n_tgt': 150},
                                        Size.LG: {'n_tgt': 200}}, init_enc()),  # Small problem sizes due to extremely large possible imputation ratios
        (AnalyticalAssignmentProblem, {Size.SM: {'n_src': 2, 'n_tgt': 4}, Size.MD: {'n_src': 3, 'n_tgt': 4},
                                       Size.LG: {'n_src': 4, 'n_tgt': 4}}, init_enc()),
        (AnalyticalAssignmentProblem, {Size.SM: {'n_src': 2, 'n_tgt': 4, 'injective': True},
                                       Size.MD: {'n_src': 5, 'n_tgt': 5, 'injective': True},
                                       Size.LG: {'n_src': 6, 'n_tgt': 6, 'injective': True}}, init_enc()),
        (AnalyticalAssignmentProblem, {Size.SM: {'n_src': 3, 'n_tgt': 4, 'injective': True, 'surjective': True},
                                       Size.MD: {'n_src': 4, 'n_tgt': 6, 'injective': True, 'surjective': True},
                                       Size.LG: {'n_src': 5, 'n_tgt': 7, 'injective': True, 'surjective': True}}, init_enc()),
        (AnalyticalAssignmentProblem, {Size.SM: {'n_src': 2, 'n_tgt': 3, 'repeatable': True},
                                       Size.MD: {'n_src': 2, 'n_tgt': 4, 'repeatable': True},
                                       Size.LG: {'n_src': 3, 'n_tgt': 4, 'repeatable': True}}, init_enc()),
        (AnalyticalPartitioningProblem, {Size.SM: {'n_src': 3, 'n_tgt': 4}, Size.MD: {'n_src': 4, 'n_tgt': 6},
                                         Size.LG: {'n_src': 5, 'n_tgt': 7}}, init_enc()),
        (AnalyticalPartitioningProblem, {Size.SM: {'n_src': 2, 'n_tgt': 4, 'covering': True},  # Same as surjective assignment problem
                                         Size.MD: {'n_src': 3, 'n_tgt': 4, 'covering': True},
                                         Size.LG: {'n_src': 4, 'n_tgt': 4, 'covering': True}}, init_enc()),
        (AnalyticalDownselectingProblem, {Size.SM: {'n_tgt': 7}, Size.MD: {'n_tgt': 12}, Size.LG: {'n_tgt': 15}}, init_enc()),
        (AnalyticalConnectingProblem, {Size.SM: {'n': 3}, Size.MD: {'n': 4}, Size.LG: None}, init_enc()),  # n = 5 --> 1M points
        (AnalyticalConnectingProblem, {Size.SM: {'n': 4, 'directed': False}, Size.MD: None,
                                       Size.LG: None}, DEFAULT_LAZY_ENCODER()),  # n >= 5 is too slow
        (AnalyticalPermutingProblem, {Size.SM: {'n': 5}, Size.MD: {'n': 7}, Size.LG: {'n': 8}}, init_enc()),
        (AnalyticalIterCombinationsProblem, {Size.SM: {'n_take': 5, 'n_tgt': 9}, Size.MD: {'n_take': 7, 'n_tgt': 15},
                                             Size.LG: {'n_take': 9, 'n_tgt': 19}}, init_enc()),
        (AnalyticalIterCombinationsReplacementProblem, {Size.SM: {'n_take': 3, 'n_tgt': 7}, Size.MD: {'n_take': 5, 'n_tgt': 10},
                                                        Size.LG: None}, init_enc()),  # Large size needs too much memory...
    ]
    if return_factories:
        return [_get_problem_factory(cls, **kwargs[size]) for cls, kwargs, enc in prob_kwargs if kwargs[size] is not None]
    return [cls(enc, **kwargs[size]) for cls, kwargs, enc in prob_kwargs if kwargs[size] is not None]


def show_problem_sizes(size: Size, reset_pf=False):
    for problem in get_problems(size):
        show_problem_size(problem)
        if reset_pf:
            problem.reset_pf_cache()
        calc_initial_hv(problem)
        print('')


def _get_recursive_encoder_factory(n_divide: int):
    def _encoder_factory(imp):
        return RecursiveEncoder(imp, n_divide=n_divide)
    return _encoder_factory


def exp_sbo_convergence(size: Size, n_repeat=8, i_prob=None, do_run=True):
    Experimenter.capture_log()
    pop_sizes = [30, 50, 80]
    n_infill = 30
    n_hv_test = [10, 20, 30]

    encoders, imputers = [], []
    assert len(EAGER_ENCODERS) == 13
    eager_encoders = EAGER_ENCODERS[:9] if size == Size.LG else EAGER_ENCODERS

    encoders += eager_encoders
    imputers += [DEFAULT_EAGER_IMPUTER]*len(eager_encoders)
    encoders += EAGER_ENUM_ENCODERS
    imputers += [DEFAULT_EAGER_IMPUTER]*len(EAGER_ENUM_ENCODERS)
    encoders += LAZY_ENCODERS
    imputers += [DEFAULT_LAZY_IMPUTER]*len(LAZY_ENCODERS)

    problems, algorithms, plot_names, n_eval_max, algo_names = [], [], [], [], []
    stats = {'prob': [], 'enc': [], 'n': [], 'n_des_space': [], 'imp_ratio': [], 'inf_idx': [],
             'n_doe': [], 'hv_doe': [], 'hv_doe_std': [], 'n_hv': []}
    for n in n_hv_test:
        stats[f'hv_{n}'] = []
        stats[f'hv_{n}_std'] = []
    stats['hv_end'] = []
    stats['hv_end_std'] = []
    i_map = {}
    i_exp = 0
    base_problems = get_problems(size)
    found = False
    for j, problem in enumerate(base_problems):
        if i_prob is not None and i_prob != j:
            continue
        found = True

        prob_encoders = []
        enc_data = {'imp_ratio': [], 'inf_idx': []}
        for i, encoder_factory in enumerate(encoders):
            encoder = encoder_factory(imputers[i]())
            # log.info(f'Encoding {problem!s} ({j+1}/{len(base_problems)}) with {encoder!s} ({i+1}/{len(encoders)})')

            enc_prob = problem.get_for_encoder(encoder)
            prob_encoders.append(enc_prob)
            enc_data['imp_ratio'].append(enc_prob.get_imputation_ratio())
            enc_data['inf_idx'].append(enc_prob.get_information_index())

        df_enc = pd.DataFrame(data=enc_data)
        i_sel = set()
        min_imp_mask = np.where(df_enc.imp_ratio == np.min(df_enc.imp_ratio.values))[0]
        i_sel.add(min_imp_mask[np.argmin(df_enc.inf_idx.values[min_imp_mask])])
        i_sel.add(min_imp_mask[np.argmax(df_enc.inf_idx.values[min_imp_mask])])
        high_inf_idx = np.where((df_enc.imp_ratio < 100) & (df_enc.inf_idx == 1))[0]

        if len(high_inf_idx) == 0:
            high_inf_idx = np.where(df_enc.imp_ratio < 100)[0]
            high_inf_idx = [high_inf_idx[np.argmax(df_enc.inf_idx.values[high_inf_idx])]]
        i_sel.add(high_inf_idx[np.argmin(df_enc.imp_ratio.values[high_inf_idx])])
        i_sel.add(high_inf_idx[np.argmax(df_enc.imp_ratio.values[high_inf_idx])])

        comb_value = (df_enc.imp_ratio.values-1)*-1 + df_enc.inf_idx.values*2
        i_sel.add(np.argmax(comb_value))

        n_valid = problem.get_n_valid_design_points()
        log.info(f'Selecting for {prob_encoders[0]!s}')
        for i in sorted(i_sel):
            enc_prob = prob_encoders[i]
            for pop_size in pop_sizes:
                stats['prob'].append(str(enc_prob))
                stats['enc'].append(str(enc_prob.assignment_manager.encoder))
                stats['n'].append(n_valid)
                stats['n_des_space'].append(enc_prob.get_n_design_points())
                stats['imp_ratio'].append(enc_prob.get_imputation_ratio())
                stats['inf_idx'].append(enc_prob.get_information_index())
                stats['n_doe'].append(pop_size)
                stats['hv_doe'].append(np.nan)
                stats['hv_doe_std'].append(np.nan)
                stats['n_hv'].append(np.nan)
                stats['hv_end'].append(np.nan)
                stats['hv_end_std'].append(np.nan)
                for n in n_hv_test:
                    stats[f'hv_{n}'].append(np.nan)
                    stats[f'hv_{n}_std'].append(np.nan)
                if pop_size == pop_sizes[0]:
                    log.info(f'Selected encoder imp_rat = {stats["imp_ratio"][-1]:.2f}; '
                             f'inf_idx = {stats["inf_idx"][-1]:.2f}; {stats["enc"][-1]}')

                problems.append(enc_prob)
                plot_names.append(f'{size.name} {enc_prob!s}: {stats["enc"][-1]!s}\n'
                                  f'imp_rat = {stats["imp_ratio"][-1]:.2f}, inf_idx = {stats["inf_idx"][-1]:.2f}\n'
                                  f'doe size = {pop_size:.0f}')
                algorithms.append(get_sbo_algo(problem, init_size=pop_size))
                n_eval_max.append(pop_size+n_infill)
                algo_names.append(f'SBO {pop_size}')

                i_map[i_exp] = len(stats['prob'])-1
                i_exp += 1
    if not found:
        return False

    exp_name = f'04_sbo_convergence_{size.value}{size.name.lower()}_sbo'

    set_results_folder(exp_name)
    res_folder = Experimenter.results_folder
    df = pd.DataFrame(data=stats)
    stats_file_post = '' if i_prob is None else f'_{i_prob}'
    df.to_csv(f'{res_folder}/stats_init{stats_file_post}.csv')

    if i_prob is not None:
        merge_csv_files(res_folder, 'stats_init', len(base_problems))

    if do_run:
        run(exp_name, problems, algorithms, algo_names=algo_names, plot_names=plot_names, n_repeat=n_repeat,
            n_eval_max=n_eval_max, do_run=do_run, do_plot=False)

    # Get and plot results
    exp = run(exp_name, problems, algorithms, algo_names=algo_names, plot_names=plot_names, n_repeat=n_repeat,
              n_eval_max=n_eval_max, return_exp=True)
    for i, experimenter in enumerate(exp):
        i_stats = i_map[i]
        agg_res = experimenter.get_aggregate_effectiveness_results()
        df.at[i_stats, 'n_hv'] = len(agg_res.metrics['delta_hv'].values['delta_hv'])
        df.at[i_stats, 'hv_doe'] = agg_res.metrics['delta_hv'].values['delta_hv'][0]
        df.at[i_stats, 'hv_doe_std'] = agg_res.metrics['delta_hv'].values_std['delta_hv'][0]
        df.at[i_stats, 'hv_end'] = agg_res.metrics['delta_hv'].values['delta_hv'][-1]
        df.at[i_stats, 'hv_end_std'] = agg_res.metrics['delta_hv'].values_std['delta_hv'][-1]
        for n in n_hv_test:
            if n-1 < len(agg_res.metrics['delta_hv'].values['delta_hv']):
                df.at[i_stats, f'hv_{n}'] = agg_res.metrics['delta_hv'].values['delta_hv'][n-1]
                df.at[i_stats, f'hv_{n}_std'] = agg_res.metrics['delta_hv'].values_std['delta_hv'][n-1]

    for n in n_hv_test:
        df[f'hv_ratio_{n}'] = df[f'hv_{n}']/df.hv_doe
        df[f'hv_ratio_{n}_std'] = df[f'hv_{n}_std']/df.hv_doe
    df['hv_ratio_end'] = df[f'hv_end']/df.hv_doe
    df['hv_ratio_end_std'] = df[f'hv_end_std']/df.hv_doe

    stats_file = f'{res_folder}/stats{stats_file_post}.csv'
    df.to_csv(stats_file)
    if i_prob is not None:
        merge_csv_files(res_folder, 'stats', len(base_problems))
    return True


def run_experiment(size: Size, sbo=False, n_repeat=8, i_prob=None, do_run=True):
    Experimenter.capture_log()
    pop_size = 30 if sbo else 50
    n_gen = 4 if size == Size.SM else 6
    n_infill = 20
    imp_ratio_limit = 500 if sbo else 1e4

    encoders, imputers = [], []

    # For large problems exclude Flat/Coord Idx location groupers
    assert len(EAGER_ENCODERS) == 13
    eager_encoders = EAGER_ENCODERS[:9] if size == Size.LG else EAGER_ENCODERS

    encoders += eager_encoders
    imputers += [DEFAULT_EAGER_IMPUTER]*len(eager_encoders)
    encoders += EAGER_ENUM_ENCODERS
    imputers += [DEFAULT_EAGER_IMPUTER]*len(EAGER_ENUM_ENCODERS)
    encoders += LAZY_ENCODERS
    imputers += [DEFAULT_LAZY_IMPUTER]*len(LAZY_ENCODERS)

    problems, algorithms, plot_names = [], [], []
    stats = {'prob': [], 'enc': [], 'n': [], 'n_des_space': [], 'imp_ratio': [], 'inf_idx': [], 'enc_time_sec': [],
             'hv_doe': [], 'hv_doe_std': [], 'hv_end': [], 'hv_end_std': []}
    i_map = {}
    i_exp = 0
    base_problems = get_problems(size)
    for j, problem in enumerate(base_problems):
        if i_prob is not None and i_prob != j:
            continue
        n_valid = problem.get_n_valid_design_points()
        for i, encoder_factory in enumerate(encoders):
            encoder = encoder_factory(imputers[i]())
            log.info(f'Encoding {problem!s} ({j+1}/{len(base_problems)}) with {encoder!s} ({i+1}/{len(encoders)})')

            s = timeit.default_timer()
            has_encoding_error = False
            try:
                with time_limiter(20.):
                    enc_prob = problem.get_for_encoder(encoder)
            except (TimeoutError, MemoryError) as e:
                log.info(f'Could not encode: {e.__class__.__name__}')
                has_encoding_error = True

            stats['enc_time_sec'].append(timeit.default_timer()-s)
            stats['prob'].append(str(enc_prob))
            stats['enc'].append(str(encoder))
            stats['n'].append(n_valid)
            stats['n_des_space'].append(enc_prob.get_n_design_points() if not has_encoding_error else np.nan)
            imp_ratio = enc_prob.get_imputation_ratio() if not has_encoding_error else np.nan
            stats['imp_ratio'].append(imp_ratio)
            stats['inf_idx'].append(enc_prob.get_information_index() if not has_encoding_error else np.nan)
            stats['hv_doe'].append(np.nan)
            stats['hv_doe_std'].append(np.nan)
            stats['hv_end'].append(np.nan)
            stats['hv_end_std'].append(np.nan)

            if has_encoding_error or imp_ratio > imp_ratio_limit:
                continue

            problems.append(enc_prob)
            plot_names.append(f'{size.name} {enc_prob!s}: {encoder!s}')
            if sbo:
                algorithms.append(get_sbo_algo(problem, init_size=pop_size))
            else:
                algorithms.append(get_ga_algo(problem, pop_size=pop_size))
            i_map[i_exp] = len(stats['prob'])-1
            i_exp += 1

    n_eval = pop_size+n_infill if sbo else pop_size*n_gen
    algo_names = ['SBO' if sbo else 'NSGA2']*len(algorithms)
    exp_name = f'04_metric_consistency_{size.value}{size.name.lower()}_{algo_names[0].lower()}'

    set_results_folder(exp_name)
    res_folder = Experimenter.results_folder
    df = pd.DataFrame(data=stats)
    stats_file_post = '' if i_prob is None else f'_{i_prob}'
    df.to_csv(f'{res_folder}/stats_init{stats_file_post}.csv')

    if i_prob is not None:
        merge_csv_files(res_folder, 'stats_init', len(base_problems))

    if do_run:
        run(exp_name, problems, algorithms, algo_names=algo_names, plot_names=plot_names, n_repeat=n_repeat,
            n_eval_max=n_eval, do_run=do_run, do_plot=False)

    # Get and plot results
    exp = run(exp_name, problems, algorithms, algo_names=algo_names, plot_names=plot_names, n_repeat=n_repeat,
              n_eval_max=n_eval, return_exp=True)
    for i, experimenter in enumerate(exp):
        i_stats = i_map[i]
        try:
            agg_res = experimenter.get_aggregate_effectiveness_results()
        except IndexError:
            log.info(f'Results not available for: {experimenter.problem.name()} / {experimenter.algorithm_name}')
            continue
        df.at[i_stats, 'hv_doe'] = agg_res.metrics['delta_hv'].values['delta_hv'][0]
        df.at[i_stats, 'hv_doe_std'] = agg_res.metrics['delta_hv'].values_std['delta_hv'][0]
        df.at[i_stats, 'hv_end'] = agg_res.metrics['delta_hv'].values['delta_hv'][-1]
        df.at[i_stats, 'hv_end_std'] = agg_res.metrics['delta_hv'].values_std['delta_hv'][-1]
    stats_file = f'{res_folder}/stats{stats_file_post}.csv'
    df.to_csv(stats_file)

    if i_prob is not None:
        df = merge_csv_files(res_folder, 'stats', len(base_problems))
    plot_stats(df, res_folder, show=not do_run)


def merge_csv_files(res_folder, filename, n):
    df_merged = None
    for j in range(n):
        csv_path = f'{res_folder}/{filename}_{j}.csv'
        if os.path.exists(csv_path):
            df_i = pd.read_csv(csv_path)
            df_merged = df_i if df_merged is None else pd.concat([df_merged, df_i], ignore_index=True)
    if df_merged is not None:
        df_merged.to_csv(f'{res_folder}/{filename}.csv')
    return df_merged


def plot_stats(df: pd.DataFrame, folder, show=False):
    df['hv_ratio'] = df['hv_end']/df['hv_doe']
    df['hv_ratio_std'] = df['hv_end_std']/df['hv_doe']
    df['hv_diff'] = df['hv_end']-df['hv_doe']
    df['hv_diff_std'] = df['hv_end_std']-df['hv_doe_std']
    for col, name in [('hv_end', 'HV end'), ('hv_ratio', 'HV ratio')]:  # , ('hv_diff', r'\delta HV')]:
        for filter_high_imp_ratio in [False]:  # [False, True]:
            x, y = df['imp_ratio'].values, df['inf_idx'].values
            z, z_std = df[col].values, df[col+'_std'].values
            keep = ~np.isnan(z)
            if filter_high_imp_ratio:
                keep &= x < 100
            x, y, z, z_std = x[keep], y[keep], z[keep], z_std[keep]

            plt.figure(), plt.title(name)
            c = plt.scatter(x, y, s=50, c=z, cmap='summer')
            plt.gca().set_xscale('log')
            plt.colorbar(c).set_label('HV (lower is better)')
            plt.xlabel('Imputation Ratio (min)'), plt.ylabel('Information Index (max)')

            fig_filename = f'{folder}/{col}'
            if filter_high_imp_ratio:
                fig_filename += '_filtered'
            plt.savefig(fig_filename+'.png')
            plt.savefig(fig_filename+'.svg')

            plt.figure(), plt.title(name)
            c = plt.scatter(x, y, s=50, c=z_std, cmap='inferno')
            plt.gca().set_xscale('log')
            plt.colorbar(c).set_label('Std dev')
            plt.xlabel('Imputation Ratio (min)'), plt.ylabel('Information Index (max)')
            plt.savefig(fig_filename+'_std.png')
            plt.savefig(fig_filename+'_std.svg')

            plt.figure(), plt.title(name)
            enc_time = df['enc_time_sec'].values[keep]
            is_lazy = np.array([enc.startswith('Lazy') for enc in df['enc'].values[keep]])
            plt.scatter(x[~is_lazy], enc_time[~is_lazy], s=20, label='Eager')
            plt.scatter(x[is_lazy], enc_time[is_lazy], s=20, label='Lazy')
            plt.gca().set_xscale('log'), plt.gca().set_yscale('log')
            plt.legend(), plt.xlabel('Imputation Ratio (min)'), plt.ylabel('Encoding time [s]')
            plt.savefig(fig_filename+'_enc_time.png')
            plt.savefig(fig_filename+'_enc_time.svg')

            plt.figure(), plt.title(name)
            c = plt.scatter(x, z, s=20, c=z_std, cmap='inferno')
            plt.gca().set_xscale('log')
            plt.colorbar(c).set_label('Std dev')
            plt.xlabel('Imputation Ratio (min)'), plt.ylabel('HV (min)')
            plt.savefig(fig_filename+'_x.png')
            plt.savefig(fig_filename+'_x.svg')

            plt.figure(), plt.title(name)
            c = plt.scatter(y, z, s=20, c=z_std, cmap='inferno')
            plt.colorbar(c).set_label('Std dev')
            plt.xlabel('Information Index (max)'), plt.ylabel('HV (min)')
            plt.savefig(fig_filename+'_y.png')
            plt.savefig(fig_filename+'_y.svg')

            plt.figure(), plt.title(name)
            c = plt.scatter(x, z, s=20, c=y, cmap='summer_r')
            plt.gca().set_xscale('log')
            plt.colorbar(c).set_label('Information Index (max)')
            plt.xlabel('Imputation Ratio (min)'), plt.ylabel('HV (min)')
            plt.savefig(fig_filename+'_x_y.png')
            plt.savefig(fig_filename+'_x_y.svg')

            plt.figure(), plt.title(name)
            c = plt.scatter(y, z, s=20, c=np.log10(x), cmap='summer')
            plt.colorbar(c).set_label('Imputation Ratio (min), log_10')
            plt.xlabel('Information Index (max)'), plt.ylabel('HV (min)')
            plt.savefig(fig_filename+'_y_x.png')
            plt.savefig(fig_filename+'_y_x.svg')

    if show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    # show_problem_sizes(Size.SM, reset_pf=True)
    # show_problem_sizes(Size.MD, reset_pf=True)
    # show_problem_sizes(Size.LG, reset_pf=True), exit()
    # show_problem_sizes(Size.SM), exit()
    # show_problem_sizes(Size.MD), exit()
    # show_problem_sizes(Size.LG), exit()

    # for ipr in list(range(len(get_problems(Size.SM)))):
    #     exp_sbo_convergence(Size.SM, n_repeat=4, i_prob=ipr)
    # for ipr in list(range(len(get_problems(Size.MD)))):
    #     exp_sbo_convergence(Size.MD, n_repeat=4, i_prob=ipr)

    run_experiment(Size.SM, n_repeat=8)
    run_experiment(Size.MD, n_repeat=8)
    # for ipr in list(range(len(get_problems(Size.LG)))):
    #     run_experiment(Size.LG, n_repeat=8, i_prob=ipr)
    # for ipr in list(range(len(get_problems(Size.SM)))):
    #     run_experiment(Size.SM, sbo=True, n_repeat=4, i_prob=ipr)
    # for ipr in list(range(len(get_problems(Size.MD)))):
    #     run_experiment(Size.MD, sbo=True, n_repeat=4, i_prob=ipr)
