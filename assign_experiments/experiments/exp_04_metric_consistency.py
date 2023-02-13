import os
import enum
import timeit
import logging
import numpy as np
import pandas as pd
from typing import *
from assign_pymoo.algo import *
from assign_pymoo.sampling import *
from assign_enc.time_limiter import *
from assign_enc.lazy_encoding import *
from assign_experiments.runner import *
from assign_enc.encoding import Encoder
from assign_enc.encoder_registry import *
from werkzeug.utils import secure_filename
from assign_pymoo.problem import AssignmentProblem
from assign_experiments.problems.analytical import *
from assign_experiments.problems.analytical_multi import *
from assign_experiments.experimenter import Experimenter
import matplotlib.pyplot as plt

log = logging.getLogger('assign_exp.exp04')

EXP4_EAGER_ENCODERS = [
    lambda imp: DirectMatrixEncoder(imp),
    lambda imp: DirectMatrixEncoder(imp, remove_gaps=False),
    lambda imp: ElementGroupedEncoder(imp),
    lambda imp: ElementGroupedEncoder(imp, normalize_within_group=False),
    lambda imp: ConnIdxGroupedEncoder(imp),
    lambda imp: ConnIdxGroupedEncoder(imp, by_src=False),

    lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), OneVarLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountGrouper(), OneVarLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountFlattenedGrouper(), OneVarLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountGrouper(), OneVarLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountFlattenedGrouper(), OneVarLocationGrouper()),

    lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), FlatIndexLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), RelFlatIndexLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountGrouper(), FlatIndexLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountFlattenedGrouper(), FlatIndexLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountGrouper(), FlatIndexLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountFlattenedGrouper(), FlatIndexLocationGrouper()),

    lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), CoordIndexLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TotalAmountGrouper(), RelCoordIndexLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountGrouper(), CoordIndexLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, SourceAmountFlattenedGrouper(), CoordIndexLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountGrouper(), CoordIndexLocationGrouper()),
    lambda imp: AmountFirstGroupedEncoder(imp, TargetAmountFlattenedGrouper(), CoordIndexLocationGrouper()),
]


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
        (AnalyticalAssignmentProblem, {Size.SM: {'n_src': 2, 'n_tgt': 4}, Size.MD: {'n_src': 3, 'n_tgt': 4},
                                       Size.LG: {'n_src': 4, 'n_tgt': 4}}, init_enc()),
        (AnalyticalAssignmentProblem, {Size.SM: {'n_src': 2, 'n_tgt': 4, 'injective': True},
                                       Size.MD: {'n_src': 5, 'n_tgt': 5, 'injective': True},
                                       Size.LG: {'n_src': 6, 'n_tgt': 6, 'injective': True}}, init_enc()),
        (AnalyticalAssignmentProblem, {Size.SM: {'n_src': 2, 'n_tgt': 3, 'repeatable': True},
                                       Size.MD: {'n_src': 2, 'n_tgt': 4, 'repeatable': True},
                                       Size.LG: {'n_src': 3, 'n_tgt': 4, 'repeatable': True}}, init_enc()),
        (AnalyticalPartitioningProblem, {Size.SM: {'n_src': 3, 'n_tgt': 4}, Size.MD: {'n_src': 4, 'n_tgt': 6},  # Same as bijective assignment problem
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


def exp_dist_corr_convergence(n_repeat=8):
    Experimenter.capture_log()
    n_dist_corr = [10, 20, 50, 100, 200, 500, 1000, 2000, -1]

    problems = [
        (_get_problem_factory(AnalyticalAssignmentProblem, **{'n_src': 2, 'n_tgt': 4, 'injective': True}), False),
        (_get_problem_factory(AnalyticalAssignmentProblem, **{'n_src': 5, 'n_tgt': 5, 'injective': True}), False),
        (_get_problem_factory(AnalyticalAssignmentProblem, **{'n_src': 2, 'n_tgt': 4, 'repeatable': True}), False),
        (_get_problem_factory(MultiPermIterCombProblem, n_take=3, n=5), True),
        (_get_problem_factory(MultiPermIterCombProblem, n_take=3, n=7), True),
    ]
    lazy_imputer = LazyFirstImputer
    encoders = [
        DirectMatrixEncoder(DEFAULT_EAGER_IMPUTER()),
        AmountFirstGroupedEncoder(DEFAULT_EAGER_IMPUTER(), TotalAmountGrouper(), OneVarLocationGrouper()),
        AmountFirstGroupedEncoder(DEFAULT_EAGER_IMPUTER(), SourceAmountGrouper(), FlatIndexLocationGrouper()),
        RecursiveEncoder(DEFAULT_EAGER_IMPUTER(), n_divide=2),
        LazyDirectMatrixEncoder(lazy_imputer()),
        LazyConnIdxMatrixEncoder(lazy_imputer(), FlatConnCombsEncoder()),
        LazyConnIdxMatrixEncoder(lazy_imputer(), FlatConnCombsEncoder(), amount_first=True),
        LazyConnIdxMatrixEncoder(lazy_imputer(), FlatConnCombsEncoder(), by_src=False),
        LazyConnIdxMatrixEncoder(lazy_imputer(), GroupedConnCombsEncoder(), amount_first=True),
        LazyAmountFirstEncoder(lazy_imputer(), SourceTargetLazyAmountEncoder(), FlatLazyConnectionEncoder()),
    ]

    stats = {'prob': [], 'enc': [], 'type': [], 'n': [], 'imp_ratio': [], 'n_dist_corr': [], 'dist_corr': [], 'dist_corr_std': [],
             'time_dist_corr': [], 'time_dist_corr_std': []}
    for i_prob, (prob_factory, min_relevant) in enumerate(problems):
        for i_enc, encoder in enumerate(encoders):
            problem = prob_factory(encoder)
            prob_enc = problem.assignment_manager.encoder
            for use_min in ([False, True] if min_relevant else [False]):
                min_str = ' (MIN)' if use_min else ''
                for n in n_dist_corr:
                    stats['prob'].append(str(problem))
                    stats['enc'].append(str(encoder))
                    stats['type'].append('min' if use_min else 'all')
                    stats['n'].append(problem.get_n_valid_design_points())
                    imp_ratio = problem.get_imputation_ratio()
                    stats['imp_ratio'].append(imp_ratio)
                    log.info(f'{i_prob+1}/{len(problems)}; {i_enc+1}/{len(encoders)}; @ {n}{min_str}: {problem!s} w/ {encoder!s} (imp ratio = {imp_ratio:.1f})')

                    dist_corr = []
                    time_dist_corr = []
                    for i_repeat in range(n_repeat):
                        s = timeit.default_timer()
                        if n == -1:
                            dist_corr_sample = prob_enc.get_distance_correlation(minimum=use_min)
                        else:
                            dv_dist, mat_dist = prob_enc._get_distance_correlation(n=n)
                            if use_min:
                                dist_corr_sample = min([prob_enc._calc_distance_corr(dvd, mat_dist[idv]) for idv, dvd in enumerate(dv_dist)])
                            else:
                                dist_corr_sample = prob_enc._calc_distance_corr(dv_dist, mat_dist)
                        time_dist_corr.append(timeit.default_timer()-s)
                        dist_corr.append(dist_corr_sample)
                        log.info(f'Distance correlation: {dist_corr_sample:.2f} ({time_dist_corr[-1]:.2g} sec, {i_repeat+1}/{n_repeat})')

                    stats['n_dist_corr'].append(n)
                    stats['dist_corr'].append(np.nanmean(dist_corr))
                    stats['dist_corr_std'].append(np.nanstd(dist_corr))
                    stats['time_dist_corr'].append(np.mean(time_dist_corr))
                    stats['time_dist_corr_std'].append(np.std(time_dist_corr))

    set_results_folder('04_dist_corr_convergence')
    res_folder = Experimenter.results_folder
    df = pd.DataFrame(data=stats)
    df.to_csv(f'{res_folder}/stats.csv')


def exp_dist_corr_perm():
    from scipy.spatial import distance
    Experimenter.capture_log()

    opt_stats = None
    for size in Size:
        exp_name = get_exp_name(size)
        set_results_folder(exp_name)
        res_folder = Experimenter.results_folder
        size_stats = pd.read_csv(f'{res_folder}/stats.csv')
        opt_stats = size_stats if opt_stats is None else pd.concat([opt_stats, size_stats], ignore_index=True)
    opt_stats = opt_stats.set_index(['prob', 'enc'])

    encoders, imputers = [], []
    encoders += EXP4_EAGER_ENCODERS
    imputers += [DEFAULT_EAGER_IMPUTER]*len(EXP4_EAGER_ENCODERS)
    encoders += EAGER_ENUM_ENCODERS
    imputers += [DEFAULT_EAGER_IMPUTER]*len(EAGER_ENUM_ENCODERS)
    encoders += LAZY_ENCODERS
    imputers += [DEFAULT_LAZY_IMPUTER]*len(LAZY_ENCODERS)

    def _calc_discrete_distance(arr):
        return distance.cdist(arr, arr, 'cityblock')  # Manhattan distance

    def _calc_perm_distance_hamming(arr):
        return distance.cdist(arr, arr, 'hamming')

    def _calc_perm_distance_idx(arr):
        from scipy.stats import kendalltau
        indices = []
        for flat_arr in arr:
            idx = np.where(flat_arr == 1)[0]
            idx -= np.arange(len(idx))*len(idx)
            indices.append(idx)
        indices = np.array(indices)
        return distance.cdist(indices, indices, 'hamming')
        # return 1-distance.cdist(indices, indices, lambda x, y: kendalltau(x, y)[0])

    stats = {'prob': [], 'enc': [], 'n': [], 'n_des_space': [], 'imp_ratio': [], 'inf_idx': [], 'dist_corr': [],
             'perm_dist_corr': [], 'hv_doe': [], 'hv_doe_std': [], 'hv_end': [], 'hv_end_std': []}
    for n in [5, 7, 8]:
        for i_enc, encoder_factory in enumerate(encoders):
            try:
                with time_limiter(10.):
                    problem = AnalyticalPermutingProblem(encoder_factory(imputers[i_enc]()), n=n)
            except TimeoutError:
                continue
            encoder = problem.assignment_manager.encoder
            imp_ratio = problem.get_imputation_ratio()
            if imp_ratio > 100:
                continue
            log.info(f'{problem!s} @ {encoder!s} ({i_enc+1}/{len(encoders)})')

            key = (str(problem), str(encoder))
            try:
                row = opt_stats.loc[key]
            except KeyError:
                continue

            stats['prob'].append(str(problem))
            stats['enc'].append(str(encoder))
            stats['n'].append(problem.get_n_valid_design_points())
            stats['n_des_space'].append(problem.get_n_design_points())
            stats['imp_ratio'].append(imp_ratio)
            stats['inf_idx'].append(problem.get_information_index())

            encoder._calc_internal_distance = _calc_discrete_distance
            stats['dist_corr'].append(encoder.get_distance_correlation())
            encoder._calc_internal_distance = _calc_perm_distance_hamming
            stats['perm_dist_corr'].append(encoder.get_distance_correlation())

            for col in ['hv_doe', 'hv_doe_std', 'hv_end', 'hv_end_std']:
                stats[col].append(row[col])

    set_results_folder('04_perm_dist_corr_type')
    res_folder = Experimenter.results_folder
    df = pd.DataFrame(data=stats)
    df.to_csv(f'{res_folder}/stats.csv')


def exp_sbo_convergence(size: Size, n_repeat=8, i_prob=None, do_run=True):
    Experimenter.capture_log()
    pop_sizes = [30, 50, 80]
    n_infill = 30
    n_hv_test = [10, 20, 30]

    encoders, imputers = [], []
    assert len(EXP4_EAGER_ENCODERS) == 23
    eager_encoders = EXP4_EAGER_ENCODERS[:11] if size == Size.LG else EXP4_EAGER_ENCODERS

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


def run_experiment(size: Size, sbo=False, n_repeat=8, i_prob=None, do_run=True, force_plot=False, force_stats=False):
    Experimenter.capture_log()
    pop_size = 30 if sbo else 50
    n_gen = 4 if size == Size.SM else 6
    n_infill = 20
    imp_ratio_limit = 100 if sbo else 500
    imp_ratio_sample_limit = 100
    n_sample_test = 100
    n_repeat_timing = (n_repeat*2) if sbo else n_repeat

    encoders, imputers = [], []

    # For large problems exclude Flat/Coord Idx location groupers
    assert len(EXP4_EAGER_ENCODERS) == 23
    eager_encoders = EXP4_EAGER_ENCODERS  # [:9] if size == Size.LG else EXP4_EAGER_ENCODERS

    encoders += eager_encoders
    imputers += [DEFAULT_EAGER_IMPUTER]*len(eager_encoders)
    encoders += EAGER_ENUM_ENCODERS
    imputers += [DEFAULT_EAGER_IMPUTER]*len(EAGER_ENUM_ENCODERS)
    encoders += LAZY_ENCODERS
    imputers += [DEFAULT_LAZY_IMPUTER]*len(LAZY_ENCODERS)

    exp_name = get_exp_name(size, sbo=sbo)
    set_results_folder(exp_name)
    res_folder = Experimenter.results_folder
    stats_file_post = '' if i_prob is None else f'_{i_prob}'
    stats_init_file = f'{res_folder}/stats_init.csv'
    stats_init_specific_file = f'{res_folder}/stats_init{stats_file_post}.csv'
    df_init_exists = None
    if not force_stats and os.path.exists(stats_init_file):
        df_init_exists = pd.read_csv(stats_init_file).set_index(['prob', 'enc'])

    problems, algorithms, plot_names = [], [], []
    stats = {'prob': [], 'enc': [], 'n': [], 'n_des_space': [], 'imp_ratio': [], 'inf_idx': [], 'dist_corr': [],
             'enc_time_sec': [], 'enc_time_sec_std': [], 'dist_corr_time_sec': [], 'dist_corr_time_sec_std': [],
             'sampling_time_sec': [], 'sampling_time_sec_std': [],
             'hv_doe': [], 'hv_doe_std': [], 'hv_end': [], 'hv_end_std': []}
    i_map = {}
    i_exp = 0
    base_problems = get_problems(size)
    for j, problem in enumerate(base_problems):
        if i_prob is not None and i_prob != j:
            continue
        n_valid = problem.get_n_valid_design_points()

        prob_enc = encoders
        prob_imp = imputers
        if isinstance(problem, AnalyticalProblemBase):
            prob_enc = [problem.get_manual_best_encoder]+prob_enc
            prob_imp = [DEFAULT_LAZY_IMPUTER]+prob_imp

        for i, encoder_factory in enumerate(prob_enc):
            encoder = encoder_factory(prob_imp[i]())
            if encoder is None:
                continue
            log.info(f'Encoding {problem!s} ({j+1}/{len(base_problems)}) with {encoder!s} ({i+1}/{len(encoders)})')

            has_encoding_error = False
            stats_key = (str(problem), str(encoder))
            is_manual_best = 'Manual Best' in str(encoder)
            if df_init_exists is not None and stats_key in df_init_exists.index:
                log.info(f'Results exist for {problem!s} (encoder: {encoder!s})')
                stats_row = df_init_exists.loc[stats_key]
                if isinstance(stats_row, pd.DataFrame):
                    if len(stats_row) != 1:
                        raise ValueError(f'Unexpected existing results for {stats_key!r}')
                    stats_row = stats_row.iloc[0, :]
                enc_prob = None
                if np.isnan(stats_row['imp_ratio']):
                    has_encoding_error = True
                else:
                    try:
                        with time_limiter(20.):
                            enc_prob = problem.get_for_encoder(encoder)
                    except (TimeoutError, MemoryError) as e:
                        log.info(f'Could not encode: {e.__class__.__name__}')
                        has_encoding_error = True

                imp_ratio = stats_row['imp_ratio']
                for col in stats:
                    if col == 'prob':
                        stats['prob'].append(str(problem))
                        continue
                    elif col == 'enc':
                        stats['enc'].append(str(encoder))
                        continue
                    elif col not in df_init_exists.columns:  # or col in ['dist_corr_time_sec', 'dist_corr_time_sec_std']:
                        if col == 'dist_corr':
                            stats[col].append(enc_prob.assignment_manager.encoder.get_distance_correlation()
                                              if not has_encoding_error and imp_ratio < imp_ratio_sample_limit else np.nan)
                            continue

                        elif col == 'dist_corr_time_sec':
                            dist_corr_times = []
                            if not has_encoding_error and imp_ratio < imp_ratio_sample_limit:
                                enc_prob.assignment_manager.encoder.get_distance_correlation()  # Once for numba jit
                                for i_test in range(n_repeat_timing):
                                    s = timeit.default_timer()
                                    enc_prob.assignment_manager.encoder.get_distance_correlation()
                                    dist_corr_times.append(timeit.default_timer() - s)

                            log.info(f'Dist corr time: {np.mean(dist_corr_times):.2g}')
                            stats[col].append(np.mean(dist_corr_times) if len(dist_corr_times) > 0 else np.nan)
                            stats[col+'_std'].append(np.std(dist_corr_times) if len(dist_corr_times) > 0 else np.nan)
                            continue
                        elif col == 'dist_corr_time_sec_std':
                            continue

                        raise RuntimeError(f'Column not found: {col}')
                    stats[col].append(stats_row[col])

            else:
                enc_times = []
                for i_test in range(n_repeat_timing):
                    try:
                        with time_limiter(20.):
                            s = timeit.default_timer()
                            enc_prob = problem.get_for_encoder(encoder)
                    except (TimeoutError, MemoryError) as e:
                        log.info(f'Could not encode: {e.__class__.__name__}')
                        has_encoding_error = True
                        enc_times.append(timeit.default_timer()-s)
                        break
                    enc_times.append(timeit.default_timer()-s)
                    log.info(f'Encoded in {enc_times[-1]:.2g} sec ({i_test+1}/{n_repeat_timing})')

                stats['prob'].append(str(problem))
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

                stats['enc_time_sec'].append(np.mean(enc_times) if len(enc_times) > 0 else np.nan)
                stats['enc_time_sec_std'].append(np.std(enc_times))
                if has_encoding_error or (imp_ratio > imp_ratio_sample_limit and not is_manual_best):
                    stats['sampling_time_sec'].append(np.nan)
                    stats['sampling_time_sec_std'].append(np.nan)
                    stats['dist_corr'].append(np.nan)
                    stats['dist_corr_time_sec'].append(np.nan)
                    stats['dist_corr_time_sec_std'].append(np.nan)
                else:
                    sampling_times = []
                    dist_corr_times = []
                    dist_corr = np.nan
                    enc_prob.assignment_manager.encoder.get_distance_correlation()  # Once for numba jit
                    for i_test in range(n_repeat_timing):
                        imputer = enc_prob.assignment_manager.encoder._imputer
                        if isinstance(imputer, LazyImputer):
                            imputer._impute_cache = {}

                        s = timeit.default_timer()
                        sampling = RepairedRandomSampling(repair=enc_prob.get_repair())
                        sampling.do(enc_prob, n_sample_test)
                        sampling_times.append((timeit.default_timer()-s)/n_sample_test)

                        s = timeit.default_timer()
                        dist_corr = enc_prob.assignment_manager.encoder.get_distance_correlation()
                        dist_corr_times.append(timeit.default_timer()-s)

                        log.info(f'Time per sample: {sampling_times[-1]:.2g} sec; '
                                 f'dist corr time: {dist_corr_times[-1]:.2f} ({i_test+1}/{n_repeat_timing})')

                    stats['sampling_time_sec'].append(np.mean(sampling_times))
                    stats['sampling_time_sec_std'].append(np.std(sampling_times))
                    stats['dist_corr'].append(dist_corr)
                    stats['dist_corr_time_sec'].append(np.mean(dist_corr_times))
                    stats['dist_corr_time_sec_std'].append(np.std(dist_corr_times))

            if has_encoding_error or (imp_ratio > imp_ratio_limit and not is_manual_best):
                continue

            problems.append(enc_prob)
            plot_names.append(f'{size.name} {enc_prob!s}: {encoder!s}')
            if sbo:
                algorithms.append(get_sbo_algo(enc_prob, init_size=pop_size))
            else:
                used_manual = False
                if isinstance(encoder, ManualBestEncoder):
                    manual_nsga2 = encoder.get_nsga2(pop_size=pop_size)
                    if manual_nsga2 is not None:
                        algorithms.append(manual_nsga2)
                        used_manual = True
                if not used_manual:
                    algorithms.append(get_ga_algo(enc_prob, pop_size=pop_size))
            i_map[i_exp] = len(stats['prob'])-1
            i_exp += 1

    n_eval = pop_size+n_infill if sbo else pop_size*n_gen
    algo_names = ['SBO' if sbo else 'NSGA2']*len(algorithms)
    df = pd.DataFrame(data=stats)
    df.to_csv(stats_init_specific_file)

    if i_prob is not None:
        merge_csv_files(res_folder, 'stats_init', len(base_problems))

    if do_run:
        run(exp_name, problems, algorithms, algo_names=algo_names, plot_names=plot_names, n_repeat=n_repeat,
            n_eval_max=n_eval, do_run=do_run, do_plot=False, only_if_needed=True)

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
    if force_plot or i_prob is None or i_prob == len(base_problems)-1:
        plot_stats(df, res_folder, show=not do_run)


def get_exp_name(size: Size, sbo=False):
    algo_name = 'SBO' if sbo else 'NSGA2'
    return f'04_metric_consistency_{size.value}{size.name.lower()}_{algo_name.lower()}'


def _do_plot(size: Size, sbo=False):
    exp_name = get_exp_name(size, sbo)
    set_results_folder(exp_name)
    res_folder = Experimenter.results_folder

    df = merge_csv_files(res_folder, 'stats', len(get_problems(size)))
    plot_stats(df, res_folder, show=False)


def merge_csv_files(res_folder, filename, n):
    df_merged = None
    for j in range(n):
        csv_path = f'{res_folder}/{filename}_{j}.csv'
        if os.path.exists(csv_path):
            df_i = pd.read_csv(csv_path)
            df_merged = df_i if df_merged is None else pd.concat([df_merged, df_i], ignore_index=True)
    merged_path = f'{res_folder}/{filename}.csv'
    if df_merged is None:
        if os.path.exists(merged_path):
            df_merged = pd.read_csv(merged_path)
        else:
            raise RuntimeError(f'Merged data not found: {merged_path}')
    else:
        df_merged.to_csv(merged_path)
    return df_merged


def plot_stats(df: pd.DataFrame, folder, show=False):
    df['hv_ratio'] = df['hv_end']/df['hv_doe']
    df['hv_ratio_std'] = df['hv_end_std']/df['hv_doe']
    df['hv_diff'] = df['hv_end']-df['hv_doe']
    df['hv_diff_std'] = df['hv_end_std']-df['hv_doe_std']

    df['has_result'] = has_result = ~np.isnan(df['hv_end'])
    df['has_timing'] = has_timing = ~np.isnan(df['sampling_time_sec'])
    df['enc_type'] = np.array([enc.startswith('Lazy') for enc in df.enc.values], dtype=float)
    df.enc_type[np.array([enc.startswith('Recursive') or enc.startswith('One Var') for enc in df.enc.values])] = .5

    col_names = {
        'n': 'Nr of valid points',
        'n_des_space': 'Nr of points in design space',
        'imp_ratio': 'Imputation ratio',
        'inf_idx': 'Information index',
        'dist_corr': 'Distance correlation',
        'enc_time_sec': 'Encoding time [s]',
        'dist_corr_time_sec': 'Distance correlation time [s]',
        'sampling_time_sec': 'Time per sample [s]',
        'hv_doe': 'HV (doe)',
        'hv_doe_std': 'HV (doe) std',
        'hv_end': 'HV (end)',
        'hv_end_std': 'HV (end) std',
        'hv_ratio': 'HV ratio = HV end/doe',
        'hv_ratio_std': 'HV ratio = HV end/doe (std)',
        'hv_diff': 'HV diff = HV end-doe',
        'enc_type': 'Encoder Type (0 = eager, .5 = enum, 1 = lazy)',
    }

    def _plot(x_col, y_col, err_col=None, z_col=None, x_log=False, y_log=False, z_log=False, xy_line=False, mask=None,
              prefix='', subtitle=None):
        x_name, y_name = col_names[x_col], col_names[y_col]
        z_name = col_names[z_col] if z_col is not None else None
        plt.figure(figsize=(8, 4))
        title = f'{x_name} vs {y_name}'
        if subtitle is not None:
            title += '\n'+subtitle
        plt.title(title)

        if mask is None:
            mask = np.ones((len(df),), dtype=bool)
        x_all, y_all, z_all, plot_color = [], [], [], False
        x, y = df[x_col].values[mask], df[y_col].values[mask]
        z = df[z_col].values[mask] if z_col is not None else None
        fmt = ''  # '--' if i >= 10 else '-'
        if err_col is not None:
            err = df[err_col].values[mask] if err_col is not None else None
            plt.errorbar(x, y, yerr=err, fmt=fmt, marker='.', markersize=5, capsize=3,
                         elinewidth=.5, linewidth=.5)
        if err_col is None or (err_col is not None and z_col is not None):
            if z_col is None:
                plt.plot(x, y, fmt, marker='.', markersize=5, linewidth=.5)
            else:
                plot_color = True
        x_all += list(x)
        y_all += list(y)
        if z_col is not None:
            z_all += list(np.log10(z) if z_log else z)

        if plot_color:
            cmap = 'viridis' if z_col == 'enc_type' else 'inferno'
            c = plt.scatter(x_all, y_all, s=50, c=z_all, cmap=cmap)
            plt.colorbar(c).set_label((z_name + ' (log)') if z_log else z_name)

        if xy_line:
            xy_min, xy_max = min(x_all+y_all), max(x_all+y_all)
            plt.plot([xy_min, xy_max], [xy_min, xy_max], '--k', linewidth=.5, label='X = Y')

        if x_log:
            plt.gca().set_xscale('log')
        if y_log:
            plt.gca().set_yscale('log')
        plt.xlabel(x_name), plt.ylabel(y_name)

        # if err_col is not None or z_col is None:
        #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        filename = f'{folder}/{prefix}{x_col}_{y_col}{f"_{z_col}" if z_col is not None else ""}' \
                   f'{f"_{err_col}" if err_col is not None else ""}'
        plt.savefig(filename+'.png')
        if not prefix:
            plt.savefig(filename+'.svg')

    for col in ['hv_end']:  # , 'hv_ratio']:
        _plot('imp_ratio', 'inf_idx', z_col=col, x_log=True, mask=has_result)
        _plot('imp_ratio', 'dist_corr', z_col=col, x_log=True, mask=has_result)
        _plot('imp_ratio', 'enc_time_sec', z_col=col, x_log=True, y_log=True, mask=has_result)
        _plot('inf_idx', 'dist_corr', z_col=col, mask=has_result)
        _plot('imp_ratio', col, z_col=col+'_std', x_log=True, mask=has_result)
        _plot('imp_ratio', col, z_col='inf_idx', x_log=True, mask=has_result)
        _plot('imp_ratio', col, z_col='dist_corr', x_log=True, mask=has_result)
        _plot('imp_ratio', col, z_col='enc_type', x_log=True, mask=has_result)
        _plot('imp_ratio', col, z_col='enc_time_sec', x_log=True, z_log=True, mask=has_result)
        _plot('inf_idx', col, z_col=col+'_std', mask=has_result)
        _plot('inf_idx', col, z_col='imp_ratio', z_log=True, mask=has_result)
        _plot('inf_idx', col, z_col='dist_corr', mask=has_result)
        _plot('inf_idx', col, z_col='enc_type', mask=has_result)
        _plot('dist_corr', col, z_col=col+'_std', mask=has_result)
        _plot('dist_corr', col, z_col='imp_ratio', z_log=True, mask=has_result)
        _plot('dist_corr', col, z_col='enc_type', mask=has_result)

    _plot('imp_ratio', 'enc_time_sec', z_col='enc_type', x_log=True, y_log=True, mask=has_timing)
    _plot('imp_ratio', 'dist_corr_time_sec', z_col='enc_type', x_log=True, y_log=True, mask=has_timing)
    _plot('imp_ratio', 'sampling_time_sec', z_col='enc_type', x_log=True, y_log=True, mask=has_timing)
    _plot('dist_corr', 'dist_corr_time_sec', z_col='enc_type', y_log=True, mask=has_timing)
    _plot('inf_idx', 'dist_corr', z_col='enc_type', mask=has_timing)
    _plot('inf_idx', 'dist_corr', z_col='imp_ratio', z_log=True, mask=has_timing)

    for i, prob in enumerate(list(df.prob.unique())):
        prob_mask = df.prob == prob
        prob_prefix = f'{i:02d}_{secure_filename(prob)}'
        kw = {'mask': prob_mask, 'prefix': prob_prefix, 'subtitle': prob}
        _plot('imp_ratio', 'inf_idx', z_col='hv_end', x_log=True, **kw)
        _plot('imp_ratio', 'dist_corr', z_col='hv_end', x_log=True, **kw)
        _plot('dist_corr', 'hv_end', z_col='imp_ratio', z_log=True, **kw)
        _plot('dist_corr', 'hv_end', z_col='enc_type', **kw)
        _plot('inf_idx', 'dist_corr', z_col='hv_end', **kw)
        _plot('inf_idx', 'dist_corr', z_col='enc_type', **kw)
        # _plot('inf_idx', 'dist_corr', z_col='imp_ratio', z_log=True, **kw)
        _plot('imp_ratio', 'enc_time_sec', z_col='enc_type', x_log=True, y_log=True, **kw)
        _plot('imp_ratio', 'sampling_time_sec', z_col='enc_type', x_log=True, y_log=True, **kw)

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

    # exp_dist_corr_convergence(), exit()
    # exp_dist_corr_perm(), exit()

    # for ipr in list(range(len(get_problems(Size.SM)))):
    #     exp_sbo_convergence(Size.SM, n_repeat=4, i_prob=ipr)
    # for ipr in list(range(len(get_problems(Size.MD)))):
    #     exp_sbo_convergence(Size.MD, n_repeat=4, i_prob=ipr)

    # _do_plot(Size.SM), _do_plot(Size.MD), _do_plot(Size.LG), exit()
    # _do_plot(Size.SM, sbo=True), _do_plot(Size.MD, sbo=True), _do_plot(Size.LG, sbo=True), exit()

    run_experiment(Size.SM, n_repeat=8)
    # run_experiment(Size.MD, n_repeat=8)
    # for ipr in list(range(len(get_problems(Size.LG)))):
    #     run_experiment(Size.LG, n_repeat=8, i_prob=ipr)
    # for ipr in list(range(len(get_problems(Size.SM)))):
    #     run_experiment(Size.SM, sbo=True, n_repeat=4, i_prob=ipr)
    # for ipr in list(range(len(get_problems(Size.MD)))):
    #     run_experiment(Size.MD, sbo=True, n_repeat=4, i_prob=ipr)
