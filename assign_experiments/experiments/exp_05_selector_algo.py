import enum
import timeit
import logging
import numpy as np
import pandas as pd
from typing import *
from assign_pymoo.algo import *
from assign_experiments.runner import *
from assign_enc.encoding import Encoder
from assign_enc.encoder_registry import *
from assign_enc.selector import EncoderSelector
from assign_pymoo.problem import AssignmentProblem
from assign_experiments.problems.gnc import GNCProblem
from assign_experiments.experimenter import Experimenter
from assign_experiments.experiments.exp_04_metric_consistency import Size, get_problems, merge_csv_files
import matplotlib.pyplot as plt

log = logging.getLogger('assign_exp.exp05')


def _get_gnc_factory(**kwargs):
    def _gnc_factory(enc):
        return GNCProblem(enc, **kwargs)
    return _gnc_factory


def get_gnc_problem_factories() -> List[Callable[[Optional[Encoder]], AssignmentProblem]]:
    factories = []
    factories += [_get_gnc_factory(choose_nr=False, n_max=n, choose_type=False) for n in [3, 4]]  # 265, 41503, [24 997 921]
    factories += [_get_gnc_factory(choose_nr=True, n_max=n, choose_type=False) for n in [3, 4]]  # 327, 46312
    factories += [_get_gnc_factory(choose_nr=False, n_max=n, choose_type=True) for n in [2, 3, 4]]  # 252, 26500, [9 338 175]
    factories += [_get_gnc_factory(choose_nr=True, n_max=n, choose_type=True) for n in [2, 3, 4]]  # 297, 29857, [10 030 642]
    return factories


def show_gnc_problem_sizes():
    for problem_factory in get_gnc_problem_factories():
        problem = problem_factory(OneVarEncoder(DEFAULT_EAGER_IMPUTER()))
        show_problem_size(problem)
        calc_initial_hv(problem)
        print('')


def get_problem_factories() -> List[Callable[[Optional[Encoder]], AssignmentProblem]]:
    factories = []
    for size in Size:
        factories += get_problems(size, return_factories=True)
    factories += get_gnc_problem_factories()
    return factories


class SelCompEffort(enum.Enum):
    VERY_LOW = 0
    LOW = 1
    MED = 2
    HIGH = 3
    VERY_HIGH = 4


def run_experiment(i_batch: int, effort: SelCompEffort, sbo=False, n_repeat=8, do_run=True):
    EncoderSelector._global_disable_cache = True
    Experimenter.capture_log()
    pop_size = 50
    n_gen = 6
    n_infill = 20
    n_batch = 4
    imp_ratio_limit = 1e4

    EncoderSelector.encoding_timeout = {
        SelCompEffort.VERY_LOW: .25,
        SelCompEffort.LOW: 2,
        SelCompEffort.MED: 2,
        SelCompEffort.HIGH: 10,
        SelCompEffort.VERY_HIGH: 10,
    }[effort]
    EncoderSelector.n_mat_max_eager = {
        SelCompEffort.VERY_LOW: 1e3,
        SelCompEffort.LOW: 1e3,
        SelCompEffort.MED: 1e4,
        SelCompEffort.HIGH: 1e4,
        SelCompEffort.VERY_HIGH: 1e5,
    }[effort]

    problems, algorithms, plot_names = [], [], []
    stats = {'prob': [], 'config': [], 'enc': [], 'n': [], 'n_des_space': [], 'imp_ratio': [], 'inf_idx': [],
             'sel_time': [], 'sel_time_std': [], 'sel_time_no_cache': [], 'sel_time_no_cache_std': [],
             'hv_doe': [], 'hv_doe_std': [], 'hv_end': [], 'hv_end_std': []}
    i_map = {}
    i_exp = 0
    i_start = i_batch*n_batch
    i_end = i_start+n_batch
    found = False
    problem_factories = get_problem_factories()
    for j, problem_factory in enumerate(problem_factories):
        if j < i_start or j >= i_end:
            continue
        found = True

        sel_times = []
        sel_times_no_cache = []
        matrix_gen = problem_factory(DEFAULT_LAZY_ENCODER()).assignment_manager.matrix_gen
        problem: Optional[AssignmentProblem] = None
        log.info(f'Encoding problem {j+1}/{len(problem_factories)} ({effort.name} effort)')
        failed = False
        for no_cache in [True, False]:
            for i in range(n_repeat):
                if no_cache:
                    matrix_gen.reset_agg_matrix_cache()
                try:
                    s = timeit.default_timer()
                    problem = problem_factory(None)
                    sel_time = timeit.default_timer()-s
                    if no_cache:
                        sel_times_no_cache.append(sel_time)
                    else:
                        sel_times.append(sel_time)
                    log.info(f'Selected best for {problem!s} in {sel_time:.2f} s ({i+1}/{n_repeat}'
                             f'{", no cache" if no_cache else ""}): {problem.assignment_manager.encoder!s}')
                except RuntimeError:
                    problem = problem_factory(DEFAULT_LAZY_ENCODER())
                    raise RuntimeError(f'Failed to select any encoder for {problem!s}!')
        if failed:
            problem = problem_factory(DEFAULT_LAZY_ENCODER())

        encoder = problem.assignment_manager.encoder

        stats['sel_time'].append(np.mean(sel_times) if not failed else np.nan)
        stats['sel_time_std'].append(np.std(sel_times) if not failed else np.nan)
        stats['sel_time_no_cache'].append(np.mean(sel_times_no_cache) if not failed else np.nan)
        stats['sel_time_no_cache_std'].append(np.std(sel_times_no_cache) if not failed else np.nan)
        stats['prob'].append(problem.get_problem_name())
        stats['config'].append(str(problem))
        stats['enc'].append(str(encoder))
        stats['n'].append(problem.assignment_manager.matrix_gen.count_all_matrices())
        stats['n_des_space'].append(encoder.get_n_design_points() if not failed else np.nan)
        imp_ratio = encoder.get_imputation_ratio() if not failed else np.nan
        stats['imp_ratio'].append(imp_ratio)
        inf_idx = encoder.get_information_index() if not failed else np.nan
        stats['inf_idx'].append(inf_idx)
        stats['hv_doe'].append(np.nan)
        stats['hv_doe_std'].append(np.nan)
        stats['hv_end'].append(np.nan)
        stats['hv_end_std'].append(np.nan)

        if not failed:
            log.info(f'Best encoder for {problem!s} = {encoder!s} (imp ratio = {imp_ratio:.2g}, inf idx = {inf_idx:.2f})')
        if failed or imp_ratio > imp_ratio_limit:
            continue

        problems.append(problem)
        plot_names.append(str(problem))
        if sbo:
            algorithms.append(get_sbo_algo(problem, init_size=pop_size))
        else:
            algorithms.append(get_ga_algo(problem, pop_size=pop_size))
        i_map[i_exp] = len(stats['prob'])-1
        i_exp += 1
    if not found:
        return False

    n_eval = pop_size+n_infill if sbo else pop_size*n_gen
    algo_names = ['SBO' if sbo else 'NSGA2']*len(algorithms)

    exp_name = get_exp_name(effort, sbo)
    set_results_folder(exp_name)
    res_folder = Experimenter.results_folder
    df = pd.DataFrame(data=stats)
    df.to_csv(f'{res_folder}/stats_init_{i_batch}.csv')
    merge_csv_files(res_folder, 'stats_init', len(problem_factories))

    if do_run:
        run(exp_name, problems, algorithms, algo_names=algo_names, plot_names=plot_names, n_repeat=n_repeat,
            n_eval_max=n_eval, do_run=do_run, do_plot=False)

    # Get and plot results
    exp = run(exp_name, problems, algorithms, algo_names=algo_names, plot_names=plot_names, n_repeat=n_repeat,
              n_eval_max=n_eval, return_exp=True)
    for i, experimenter in enumerate(exp):
        i_stats = i_map[i]
        agg_res = experimenter.get_aggregate_effectiveness_results()
        df.at[i_stats, 'hv_doe'] = agg_res.metrics['delta_hv'].values['delta_hv'][0]
        df.at[i_stats, 'hv_doe_std'] = agg_res.metrics['delta_hv'].values_std['delta_hv'][0]
        df.at[i_stats, 'hv_end'] = agg_res.metrics['delta_hv'].values['delta_hv'][-1]
        df.at[i_stats, 'hv_end_std'] = agg_res.metrics['delta_hv'].values_std['delta_hv'][-1]
    stats_file = f'{res_folder}/stats_{i_batch}.csv'
    df.to_csv(stats_file)
    df = merge_csv_files(res_folder, 'stats', len(problem_factories))
    plot_stats(df, res_folder, effort)
    return True


def get_exp_name(effort: SelCompEffort, sbo=False):
    algo_name = 'SBO' if sbo else 'NSGA2'
    return f'05_selector_algo_{effort.value}{effort.name.lower()}_{algo_name.lower()}'


def _do_plot(effort: SelCompEffort, sbo=False):
    exp_name = get_exp_name(effort, sbo)
    set_results_folder(exp_name)
    res_folder = Experimenter.results_folder

    df = merge_csv_files(res_folder, 'stats', len(get_problem_factories()))
    plot_stats(df, res_folder, effort, show=True)


def plot_stats(df: pd.DataFrame, folder, effort, show=False):
    # Fill in invalid values
    selection_failed = np.isnan(df.imp_ratio.values)
    df.n_des_space[selection_failed] = df.n[selection_failed]*1e4
    df.imp_ratio[selection_failed] = 1e4
    df.inf_idx[selection_failed] = 0
    df.sel_time[selection_failed] = 20
    df.sel_time_std[selection_failed] = 0
    df['is_failed'] = selection_failed
    failure_rate = sum(selection_failed)/len(selection_failed)

    no_opt_res = np.isnan(df.hv_end)
    df.hv_doe[no_opt_res] = 1.5
    df.hv_doe_std[no_opt_res] = 0
    df.hv_end[no_opt_res] = 1.5
    df.hv_end_std[no_opt_res] = 0

    df['is_lazy'] = np.array([enc.startswith('Lazy') for enc in df.enc.values], dtype=float)
    df.is_lazy[selection_failed] = .5

    # Separate by problem
    masks, names = [], []
    for prob_name in np.unique(df.prob.values):
        masks.append(df.prob.values == prob_name)
        names.append(prob_name)

    col_names = {
        'n': 'Nr of valid points',
        'n_des_space': 'Nr of points in design space',
        'imp_ratio': 'Imputation ratio',
        'inf_idx': 'Information index',
        'sel_time': 'Encoder selection time [s]',
        'sel_time_no_cache': 'Encoder selection time (no matrix cache) [s]',
        'hv_doe': 'HV (doe)',
        'hv_end': 'HV (end)',
        'is_lazy': 'Is Lazy Encoder (0.5 = selection failed)',
    }

    def _plot(x_col, y_col, err_col=None, z_col=None, x_log=False, y_log=False, z_log=False, xy_line=False):
        x_name, y_name = col_names[x_col], col_names[y_col]
        z_name = col_names[z_col] if z_col is not None else None
        plt.figure(figsize=(8, 4))
        title = f'{x_name} vs {y_name}{"" if z_name is None else f" ({z_name})"}\n{effort.name} effort'
        if z_col == 'is_lazy':
            title += f'\nFailure rate: {failure_rate*100:.0f}%'
        plt.title(title)

        x_all, y_all, z_all, plot_color = [], [], [], False
        for i, name in enumerate(names):
            mask = masks[i]
            x, y = df[x_col].values[mask], df[y_col].values[mask]
            z = df[z_col].values[mask] if z_col is not None else None
            fmt = '--' if i >= 10 else '-'
            if err_col is not None:
                err = df[err_col].values[mask] if err_col is not None else None
                plt.errorbar(x, y, yerr=err, fmt=fmt, marker='.', markersize=5, capsize=3,
                             elinewidth=.5, linewidth=.5, label=name)
            if err_col is None or (err_col is not None and z_col is not None):
                if z_col is None:
                    plt.plot(x, y, fmt, marker='.', markersize=5, linewidth=.5, label=name)
                else:
                    plot_color = True
            x_all += list(x)
            y_all += list(y)
            if z_col is not None:
                z_all += list(np.log10(z) if z_log else z)

        if plot_color:
            c = plt.scatter(x_all, y_all, s=50, c=z_all, cmap='inferno')
            plt.colorbar(c).set_label((z_name + ' (log)') if z_log else z_name)

        if xy_line:
            xy_min, xy_max = min(x_all+y_all), max(x_all+y_all)
            plt.plot([xy_min, xy_max], [xy_min, xy_max], '--k', linewidth=.5, label='X = Y')

        if x_log:
            plt.gca().set_xscale('log')
        if y_log:
            plt.gca().set_yscale('log')
        plt.xlabel(x_name), plt.ylabel(y_name)

        if err_col is not None or z_col is None:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        filename = f'{folder}/{x_col}_{y_col}{f"_{z_col}" if z_col is not None else ""}' \
                   f'{f"_{err_col}" if err_col is not None else ""}'
        plt.savefig(filename+'.png')
        plt.savefig(filename+'.svg')

    _plot('n', 'imp_ratio', x_log=True, y_log=True)
    _plot('n', 'imp_ratio', z_col='is_lazy', x_log=True, y_log=True)
    _plot('n', 'inf_idx', x_log=True)
    _plot('n', 'inf_idx', z_col='is_lazy', x_log=True)
    _plot('n', 'sel_time', err_col='sel_time_std', x_log=True, y_log=True)
    _plot('n', 'sel_time_no_cache', err_col='sel_time_no_cache_std', x_log=True, y_log=True)
    _plot('n', 'hv_end', err_col='hv_end_std', x_log=True)

    _plot('imp_ratio', 'inf_idx', z_col='sel_time', x_log=True, z_log=True)
    _plot('imp_ratio', 'inf_idx', z_col='sel_time_no_cache', x_log=True, z_log=True)
    _plot('imp_ratio', 'inf_idx', z_col='is_lazy', x_log=True)
    _plot('imp_ratio', 'inf_idx', z_col='hv_end', x_log=True)
    _plot('imp_ratio', 'inf_idx', z_col='hv_end', x_log=True)
    _plot('imp_ratio', 'sel_time', x_log=True, y_log=True)
    _plot('imp_ratio', 'sel_time_no_cache', x_log=True, y_log=True)
    _plot('sel_time', 'sel_time_no_cache', x_log=True, y_log=True, xy_line=True)

    if show:
        plt.show()
    plt.close('all')


def plot_selector_areas():
    import matplotlib.cm
    import matplotlib.patches as patches
    set_results_folder(get_exp_name(SelCompEffort.LOW))
    res_folder = Experimenter.results_folder

    cmap = matplotlib.cm.get_cmap('summer')

    plt.figure(), plt.title('Selector priority areas\n* = only if eager encoders have also been checked')
    ax = plt.gca()

    def _add_area(i_area, x, y, only_if_needed=False):
        color = cmap((i_area-1)/(n_areas-1))
        ax.add_patch(patches.Rectangle((x[0], y[0]), x[1]-x[0], y[1]-y[0], facecolor=color))
        plt.text(10**(.5*(np.log10(x[0])+np.log10(x[1]))), .5*(y[0]+y[1]), str(i_area)+('*' if only_if_needed else ''),
                 horizontalalignment='center', verticalalignment='center')

    min_inf_idx = EncoderSelector.min_information_index
    imp_rat = EncoderSelector.imputation_ratio_limits
    n_areas = 3*(len(imp_rat)+2)
    max_imp_ratio_plot = imp_rat[-1]*10

    _plot_imp_ratio_left = [.9, 1.1]
    _add_area(1, _plot_imp_ratio_left, [min_inf_idx, 1])
    _add_area(3, _plot_imp_ratio_left, [.5*min_inf_idx, min_inf_idx])
    _add_area(5, _plot_imp_ratio_left, [0, .5*min_inf_idx], only_if_needed=True)

    _add_area(2, [_plot_imp_ratio_left[1], imp_rat[0]], [min_inf_idx, 1])
    _add_area(4, [_plot_imp_ratio_left[1], imp_rat[0]], [.5*min_inf_idx, min_inf_idx])
    _add_area(6, [_plot_imp_ratio_left[1], imp_rat[0]], [0, .5*min_inf_idx], only_if_needed=True)

    imp_rat_border = imp_rat+[max_imp_ratio_plot]
    nr = 7
    for i in range(len(imp_rat)):
        _add_area(nr, [imp_rat_border[i], imp_rat_border[i+1]], [min_inf_idx, 1], only_if_needed=True)
        _add_area(nr+1, [imp_rat_border[i], imp_rat_border[i+1]], [.5*min_inf_idx, min_inf_idx], only_if_needed=True)
        _add_area(nr+2, [imp_rat_border[i], imp_rat_border[i+1]], [0, .5*min_inf_idx], only_if_needed=True)
        nr += 3

    ax.set_xscale('log')
    plt.xlim([_plot_imp_ratio_left[0], max_imp_ratio_plot]), plt.ylim([0, 1])
    plt.xlabel('Imputation ratio'), plt.ylabel('Information index')

    plt.savefig(f'{res_folder}/selector_areas.png')
    plt.savefig(f'{res_folder}/selector_areas.svg')
    plt.show()


if __name__ == '__main__':
    # show_gnc_problem_sizes(), exit()

    # plot_selector_areas(), exit()
    # _do_plot(SelCompEffort.LOW), exit()
    for eft in list(SelCompEffort)[:3]:
        for ib in list(range(30)):
            if not run_experiment(ib, eft, n_repeat=8):
                break
