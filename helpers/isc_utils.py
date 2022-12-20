from itertools import combinations

# plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# ISC
from brainiak.isc import (_check_targets_input, _check_timeseries_input,
                          _threshold_nans, bootstrap_isc, compute_correlation,
                          compute_summary_statistic, isc, isfc,
                          squareform_isfc)
from scipy.spatial.distance import squareform
from scipy.stats import norm, permutation_test, zscore
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams["errorbar.capsize"] = 0.5

import cmasher as cmr  # CITE ITS PAPER IN YOUR MANUSCRIPT


def set_plot_ticks(args, ax):
    try:
        ax.set_yticks(args.major_ticks, args.major_tick_labels, rotation=0, va='center')
        ax.set_xticks(args.major_ticks, args.major_tick_labels, rotation=90, ha='center')

        ax.set_yticks(args.minor_ticks-0.5, minor=True)
        ax.set_xticks(args.minor_ticks-0.5, minor=True)
        ax.tick_params(
            which='major', direction='out', length=5.5, 
            # grid_color='white', grid_linewidth='1.5',
            labelsize=10,
        )
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1.5)
    except:
        pass


# DATA TIME SERIES
# -----------------
def get_block_time_series(args, X, ):
    # step 1. sort trials based on number of nan values present
    for label in args.LABELS:
        for idx in np.arange(len(X[label])):
            x = X[label][idx]
            num_nans_trial = np.squeeze(
                np.apply_over_axes(
                    np.sum, 
                    np.isnan(x), 
                    axes=(1, 2)
                )
            )
            num_nans_trial_idxs = np.argsort(num_nans_trial)
            x = x[num_nans_trial_idxs, :, :]
            X[label][idx] = x

    # step 2. create time series
    '''
    find minimum number of trials across subjects
    '''
    min_trials = []
    for label in args.LABELS:
        min_trials += [x.shape[0] for x in X[label]]
    min_trials = min(min_trials)
    print(f"minimum number of trials = {min_trials}")

    '''
    time series of early and late periods
    '''
    args.PERIODS = ['early', 'late']
    # time periods
    args.TR = 1.25 #seconds
    EARLY = np.arange(2.5, 8.75+args.TR, args.TR) // args.TR
    LATE = np.arange(10.0, 16.25+args.TR, args.TR) // args.TR
    EARLY = EARLY.astype(int)
    LATE = LATE.astype(int)
    args.PERIOD_TRS = [EARLY, LATE]

    ts = {}
    for label, name in zip(args.LABELS, args.NAMES):
        for idx_period, (period, TRs) in enumerate(zip(args.PERIODS, args.PERIOD_TRS)):
            ts[f"{name}_{period}"] = []
            for x in X[label]:
                x = x[:, args.PERIOD_TRS[idx_period], :]
                trl, t, r = x.shape
                x = np.reshape(x[:min_trials, ...], (min_trials*t, r))
                ts[f"{name}_{period}"] += [zscore(x, axis=0, nan_policy='omit')]

    for block in ts.keys():
        ts[block] = np.dstack(ts[block])
    
    return ts

def get_max_block_time_series(args, X, ):
    return get_block_time_series(args, X,)

def get_aba_block_time_series(args, X):
    '''
    find minimum number of trials across subjects
    '''
    min_trials = []
    for idx, _ in enumerate(args.LABEL_NAMES):
        min_trials += [x.shape[0] for x in X[idx]]
    min_trials = min(min_trials)
    print(f"minimum number of trials = {min_trials}")

    '''
    time series for the late period
    '''
    args.TR = 1.25
    args.LATE_PERIOD_TRS = (np.arange(-3.75, 1.25+args.TR, args.TR) // args.TR + 8.0).astype(int)
    # because play block/trial starts at -8TR and ends at 4TR, and play period ends at 0TR.
    ts = {}
    for label, name in enumerate(args.LABEL_NAMES):
        ts[f"{name}"] = []
        for x in X[label]:
            x = x[:, args.LATE_PERIOD_TRS, :]
            trl, t, r = x.shape
            x = np.reshape(x[:min_trials, ...], (min_trials*t, r))
            ts[f"{name}"] += [zscore(x, axis=0, nan_policy='omit')]
    
    for block in ts.keys():
        ts[block] = np.dstack(ts[block])
    
    return ts

def get_emo2_segment_time_series(args, X):
    '''
    find minimum number of trials across subjects
    '''
    min_trials = []
    for idx, _ in enumerate(args.LABELS):
        min_trials += [x.shape[0] for x in X[idx]]
    min_trials = min(min_trials)
    print(f"minimum number of trials = {min_trials}")

    '''
    time series for APPR and RETR segments
    '''
    ts = {}
    for idx_label, (label, name) in enumerate(zip(args.LABELS, args.LABEL_NAMES)):
        ts[f"{name}"] = []
        for x in X[idx_label]:
            ts[f"{name}"] += [zscore(x[:min_trials, :], axis=0, nan_policy='omit')]
        
    for block in ts.keys():
        ts[block] = np.dstack(ts[block])

    return ts

# ISC/ISFC MATRICES
# -----------------
def get_isfcs(args, ts, print_stats=False):
    corrs = {}; bootstraps = {}; rois = {}

    for block in tqdm(ts.keys()):
        # if block == 'safe_early': continue
        isfcs, iscs = isfc(
            ts[block], 
            pairwise=args.pairwise, 
            summary_statistic=None,
            vectorize_isfcs=args.vectorize_isfcs,
            positive_only=args.positive_only
        )
        corrs[block] = {'isfcs':isfcs, 'iscs':iscs}

        bootstraps[block] = {}
        rois[block] = {}
        for corr_name in args.CORR_NAMES:
            
            # permutation test
            observed, ci, p, distribution = bootstrap_isc(
                corrs[block][corr_name], 
                pairwise=args.pairwise, 
                summary_statistic='median',
                n_bootstraps=args.n_bootstraps,
                ci_percentile=95, 
                side='two-sided',
                random_state=args.SEED
            )
            bootstraps[block][corr_name] = observed, ci, p, distribution

            # surviving rois
            # rois[block][corr_name] = q[block][corr_name] < 0.05
            rois[block][corr_name] = bootstraps[block][corr_name][2] < 0.05 # p < 0.05

            # correlations only for surviving rois
            # corrs[block][corr_name] *= rois[block][corr_name]

            if print_stats == True:
                print(
                    (
                        f"condition {block} and correlation {corr_name[:-1]} : " 
                        f"{100. * np.sum(rois[block][corr_name]) / len(rois[block][corr_name])} %"
                        f"significant roi(-pairs)"
                    )
                )

    return corrs, bootstraps, rois

def get_squareform_matrices(args, bootstraps, rois, threshold_mats=True):
    observed_isfcs = {}; observed_p_vals = {}; 
    significant_rois = {}; conf_intervals = {}
    for block in bootstraps.keys():
        # if block == 'safe_early': continue
        observed_isfcs[block] = squareform_isfc(
            bootstraps[block]['isfcs'][0], 
            bootstraps[block]['iscs'][0]
        )
        observed_p_vals[block] = squareform_isfc(
            bootstraps[block]['isfcs'][2], 
            bootstraps[block]['iscs'][2]
        )
        significant_rois[block] = squareform_isfc(
            rois[block]['isfcs'], 
            rois[block]['iscs']
        )
        conf_intervals[block] = (
            squareform_isfc(
                bootstraps[block]['isfcs'][1][0],
                bootstraps[block]['iscs'][1][0]
            ),
            squareform_isfc(
                bootstraps[block]['isfcs'][1][1],
                bootstraps[block]['iscs'][1][1]
            )
        )

        if threshold_mats == True:
            observed_isfcs[block] *= significant_rois[block]
    
    return observed_isfcs, observed_p_vals, significant_rois, conf_intervals

def get_bootstrap_distribution_isfcs(args, observed_isfcs, bootstraps):
    bootstrap_isfcs = {}
    all_isfcs = {}
    for block in bootstraps.keys():
        bootstrap_isfcs[block] = np.concatenate(
            [
                bootstraps[block]['isfcs'][3], 
                bootstraps[block]['iscs'][3]
            ], 
            axis=-1
        )

        all_isfcs[block] = np.concatenate(
            [
                np.concatenate(squareform_isfc(observed_isfcs[block]), axis=-1)[None, :],
                bootstrap_isfcs[block]
            ],
            axis=0
        )
    return bootstrap_isfcs, all_isfcs

def separate_pos_neg_weights(args, all_isfcs, significant_rois, threshold_mats=True):
    all_isfcs_pos = {}; all_isfcs_neg = {}
    all_sq_isfcs_pos = {}; all_sq_isfcs_neg = {}
    for block in all_isfcs.keys():
        all_isfcs_pos[block] = np.multiply(
            all_isfcs[block] > 0,
            all_isfcs[block]
        )

        all_isfcs_neg[block] = -1 * np.multiply(
            all_isfcs[block] < 0,
            all_isfcs[block]
        )  

        all_sq_isfcs_pos[block] = np.zeros((args.num_rois, args.num_rois, args.n_bootstraps+1))
        all_sq_isfcs_neg[block] = np.zeros((args.num_rois, args.num_rois, args.n_bootstraps+1))
        for idx_bs in np.arange(len(all_isfcs_pos[block])):
            all_sq_isfcs_pos[block][:, :, idx_bs] = squareform_isfc(
                all_isfcs_pos[block][idx_bs][:-args.num_rois],
                all_isfcs_pos[block][idx_bs][args.num_rois:]
            )

            all_sq_isfcs_neg[block][:, :, idx_bs] = squareform_isfc(
                all_isfcs_neg[block][idx_bs][:-args.num_rois],
                all_isfcs_neg[block][idx_bs][args.num_rois:]
            )
    
    return all_isfcs_pos, all_isfcs_neg, all_sq_isfcs_pos, all_sq_isfcs_neg

def get_min_max(d, corr_type='isfc'):
    '''
    min and max values of the matrices:
    used in plotting the matrices
    '''
    vals = []
    for block in d.keys():
        vals.append(d[block])
    vals = np.concatenate(vals, axis=0).flatten()
    vmin = np.nanquantile(vals, q=0.05)
    vmax = np.nanquantile(vals, q=0.95)
    if corr_type == 'isfc':
        return -max(-vmin, vmax), max(-vmin, vmax)
    elif corr_type == 'fc':
        return vmin, vmax

def plot_isfcs(args, isfcs, rois): 
    vmin, vmax = get_min_max(isfcs)

    nrows, ncols = len(args.LABELS), len(args.PERIODS)
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(5*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=0.65, hspace=0.65
    )

    for label, name in zip(args.LABELS, args.NAMES):
        for idx_period, period in enumerate(args.PERIODS):
            ax = axs[label, idx_period]
            block = f"{name}_{period}"

            # if block == 'safe_early': continue

            im = ax.imshow(
                isfcs[block], #* rois[block], 
                cmap=cmr.iceburn, vmin=vmin, vmax=vmax
            )
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if label == 0: ax.set_title(f"{period}")
            if idx_period == 0: ax.set_ylabel(f"{name}", size='large')
            
            ax.set_yticks(args.major_ticks, args.major_tick_labels, rotation=0, va='center')
            ax.set_xticks(args.major_ticks, args.major_tick_labels, rotation=90, ha='center')

            ax.set_yticks(args.minor_ticks-0.5, minor=True)
            ax.set_xticks(args.minor_ticks-0.5, minor=True)
            ax.tick_params(
                which='major', direction='out', length=5.5, 
                # grid_color='white', grid_linewidth='1.5',
                labelsize=10,
            )
            ax.grid(which='minor', color='w', linestyle='-', linewidth=1.5)

    return None

def plot_max_isfcs(args, isfcs, rois, cmap=cmr.iceburn, corr_type='isfc'):
    vmin, vmax = get_min_max(isfcs, corr_type)

    nrows, ncols = len(args.NAMES), len(args.PERIOD_TRS)
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(5*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=0.65, hspace=0.65
    )

    for label, name in enumerate(args.NAMES):
        for idx_period, period in enumerate(args.PERIOD_TRS.keys()):
            ax = axs[label, idx_period]
            block = f"{name}_{period}"

            im = ax.imshow(
                isfcs[block], #* rois[block], 
                cmap=cmap, vmin=vmin, vmax=vmax
            )
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if label == 0: ax.set_title(f"{period}")
            if idx_period == 0: ax.set_ylabel(f"{name}", size='large')

            set_plot_ticks(args, ax)


def plot_aba_isfcs(args, isfcs, rois, cmap=cmr.iceburn, corr_type='isfc'):
    vmin, vmax = get_min_max(isfcs, corr_type)

    nrows, ncols = [len(args.NAMES)//2]*2
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(5*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=0.5, hspace=0.5
    )

    for idx_valence, valence in enumerate(args.VALENCE):
        for idx_level, level in enumerate(args.LEVELS):
            ax = axs[idx_valence, idx_level]

            block = f"{level}_{valence}"

            im = ax.imshow(
                isfcs[block][np.ix_(args.roi_idxs, args.roi_idxs)], 
                #* roi[block],
                cmap=cmap, vmin=vmin, vmax=vmax
            )
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if idx_valence == 0: ax.set_title(f"{level}")
            if idx_level == 0: ax.set_ylabel(f"{valence}", size='large')

            set_plot_ticks(args, ax)


def plot_emo2_isfcs(args, isfcs, rois):
    vmin, vmax = get_min_max(isfcs)

    nrows, ncols = 1, len(args.LABEL_NAMES)
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(5*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=0.65, hspace=None
    )

    for idx, (label, name) in enumerate(zip(args.LABELS, args.LABEL_NAMES)):
        ax = axs[idx]
        block = name

        im = ax.imshow(
            isfcs[block], #* roi[block],
            cmap=cmr.iceburn, vmin=vmin, vmax=vmax
        )
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(block)

        ax.set_yticks(args.major_ticks, args.major_tick_labels, rotation=0, va='center')
        ax.set_xticks(args.major_ticks, args.major_tick_labels, rotation=90, ha='center')

        ax.set_yticks(args.minor_ticks-0.5, minor=True)
        ax.set_xticks(args.minor_ticks-0.5, minor=True)
        ax.tick_params(
            which='major', direction='out', length=5.5, 
            # grid_color='white', grid_linewidth='1.5',
            labelsize=10,
        )
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1.5)

    return None

def plot_mashid_isfcs(args, isfcs, rois):
    vmin, vmax = get_min_max(isfcs)

    nrows, ncols = 7, len(args.LABEL_NAMES)
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(5*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=0.4, hspace=0.6
    )

    for block in isfcs.keys():
        tr = np.int(block.split('_')[1][-1])
        name = block.split('_')[0]
        
        if name == 'RETR':
            ax = axs[tr, 0]
        elif name == 'APPR':
            ax = axs[tr, 1]

        im = ax.imshow(
            isfcs[block], #* roi[block],
            cmap=cmr.iceburn, vmin=vmin, vmax=vmax
        )
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(name)

        ax.set_ylabel(f"TR{tr}")

        ax.set_yticks(args.major_ticks, args.major_tick_labels, rotation=0, va='center')
        ax.set_xticks(args.major_ticks, args.major_tick_labels, rotation=90, ha='center')

        ax.set_yticks(args.minor_ticks-0.5, minor=True)
        ax.set_xticks(args.minor_ticks-0.5, minor=True)
        ax.tick_params(
            which='major', direction='out', length=5.5, 
            # grid_color='white', grid_linewidth='1.5',
            labelsize=10,
        )
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1.5)

    return None

def plot_emo2_near_miss_isfcs(args, isfcs, rois, cmap=cmr.iceburn, corr_type='isfc'):
    vmin, vmax = get_min_max(isfcs, corr_type)

    nrows, ncols = 1, len(args.SEGMENT_TRS)
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(5*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=None, hspace=0.65
    )

    for idx, tr in enumerate(args.SEGMENT_TRS):
        ax = axs[idx]
        block = tr

        im = ax.imshow(
            isfcs[block], #* roi[block],
            cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f"TR: {block}")

        set_plot_ticks(args, ax)

    return None

def plot_emo2_near_vs_far_isfcs(args, isfcs, rois, cmap=cmr.iceburn, corr_type='isfc'):
    vmin, vmax = get_min_max(isfcs, corr_type)

    nrows, ncols = 1, len(isfcs.keys())
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(5*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=None, hspace=0.65
    )

    for idx, block in enumerate(isfcs.keys()):
        ax = axs[idx]

        im = ax.imshow(
            isfcs[block], #* roi[block],
            cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f"{block}")

        set_plot_ticks(args, ax)

    return None


# COMPARISONS BETWEEN ISC MATRICES
# --------------------------
def get_comparison_stats(args, corrs, paradigm='max', how_pairs='all_combs'):
    def statistic(obs1, obs2, axis):
        return (
            + compute_summary_statistic(obs1, summary_statistic='median', axis=axis) 
            - compute_summary_statistic(obs2, summary_statistic='median', axis=axis)
        ) 

    stats_results = {}
    if paradigm in ['emo2_near_miss']:
        blocks = list(corrs.keys())[::-1]
    else:
        blocks = list(corrs.keys())

    if how_pairs == 'all_combs':
        pairs = combinations(blocks, 2)
    elif how_pairs == 'consecutive':
        pairs = zip(blocks[:-1], blocks[1:])

    for (block1, block2) in tqdm(pairs):
        obs1 = np.concatenate([corrs[block1]['isfcs'], corrs[block1]['iscs']], axis=-1)
        obs2 = np.concatenate([corrs[block2]['isfcs'], corrs[block2]['iscs']], axis=-1)
        
        stats_results[(block1, block2)] = permutation_test(
            data=(obs1, obs2),
            statistic=statistic,
            permutation_type='samples',
            vectorized=True,
            n_resamples=args.n_bootstraps,
            batch=1,
            alternative='two-sided',
            axis=0,
            random_state=args.SEED
        )
    
    return stats_results

def get_diff_isfcs(args, stats_results, significant_rois, threshold_mats=True):
    diff_isfcs = {}; diff_pvals = {}
    for (block1, block2) in stats_results.keys():

        diff_isfcs[(block1, block2)] = squareform_isfc(
            isfcs=stats_results[(block1, block2)].statistic[:-args.num_rois],
            iscs=stats_results[(block1, block2)].statistic[-args.num_rois:]
        )

        diff_pvals[(block1, block2)] = squareform_isfc(
            isfcs=stats_results[(block1, block2)].pvalue[:-args.num_rois],
            iscs=stats_results[(block1, block2)].pvalue[-args.num_rois:],
        )
        diff_pvals[(block1, block2)] = diff_pvals[(block1, block2)] < 0.05
        # FDR correction
        # coming soon...

        # keep isfc values only for significant roi pairs
        if threshold_mats == True:
            # diff_isfcs[(block1, block2)] *= diff_pvals[(block1, block2)]
            diff_isfcs[(block1, block2)] *= significant_rois[block1]
            diff_isfcs[(block1, block2)] *= significant_rois[block2]

    return diff_isfcs, diff_pvals

def plot_isfc_comparisons(args, corrs, diff_isfcs, paradigm='max'):
    vmin, vmax = get_min_max(diff_isfcs)

    nrows, ncols = [len(corrs.keys())]*2
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(4*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=0.8, hspace=0.0
    )

    if paradigm in ['emo2_near_miss']:
        blocks = list(corrs.keys())[::-1]
    else:
        blocks = list(corrs.keys())

    for idx_blk1, block1 in enumerate(blocks):
        for idx_blk2, block2 in enumerate(blocks):

            if idx_blk1 >= idx_blk2: 
                axs[idx_blk1, idx_blk2].remove()
                continue

            ax = axs[idx_blk1, idx_blk2]

            im = ax.imshow(
                diff_isfcs[(block1, block2)], 
                cmap=cmr.iceburn, vmin=vmin, vmax=vmax
            )
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if idx_blk1 == 0: ax.set_title(f"{block2}")
            if idx_blk2 == idx_blk1+1: ax.set_ylabel(f"{block1}", size='large')

            set_plot_ticks(args, ax)

    # fig.tight_layout()

def plot_max_isfc_comparisons(args, corrs, diff_isfcs):
    plot_isfc_comparisons(args, corrs, diff_isfcs, 'max')

def plot_aba_isfc_comparisons(args, corrs, diff_isfcs):
    plot_isfc_comparisons(args, corrs, diff_isfcs, 'aba')

def plot_emo2_isfc_comparisons(args, corrs, diff_isfcs):
    vmin, vmax = get_min_max(diff_isfcs)

    nrows, ncols = 1, 1
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(4*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=0.8, hspace=0.0
    )

    ax = axs

    for (blk1, blk2) in diff_isfcs.keys():
        im = ax.imshow(
            diff_isfcs[(blk1, blk2)],
            cmap=cmr.iceburn, vmin=vmin, vmax=vmax
        )
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f"{blk1} - {blk2}")

        set_plot_ticks(args, ax)

    return None

def plot_emo2_near_miss_isfc_comparisons(args, corrs, diff_isfcs, paradigm='max'):
    vmin, vmax = get_min_max(diff_isfcs)

    nrows, ncols = 1, len(corrs.keys())
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(4*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=0.8, hspace=0.0
    )

    if paradigm in ['max', 'emo2_near_miss']:
        blocks = list(corrs.keys())[::-1]
    else:
        blocks = list(corrs.keys())

    for idx_blk1, block1 in enumerate(blocks):
        for idx_blk2, block2 in enumerate(blocks):

            if idx_blk1 != idx_blk2-1: 
                continue

            ax = axs[idx_blk1]

            im = ax.imshow(
                diff_isfcs[(block1, block2)], 
                cmap=cmr.iceburn, vmin=vmin, vmax=vmax
            )
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(f"TR: {block2} - TR: {block1}")

            set_plot_ticks(args, ax)
    
    # fig.tight_layout()


# NETWORK WEIGHT PLOTS
# --------------------------
def combine_observed_and_bootstrap_isfcs(observed_isfcs, bootstrap_isfcs):
    all_isfcs = bootstrap_isfcs.copy()
    for block in all_isfcs.keys():
        all_isfcs[block] = np.concatenate(
            [observed_isfcs[block][:, :, None], bootstrap_isfcs[block]],
            axis=-1
        )
        # first matrix is the observed ISC
    return all_isfcs

def get_nw_weights(args, iscs,):
    def get_ts(args, iscs, nw1, nw2, idx_bs):
        wts = []
        for block in iscs.keys():
            vals = iscs[block][
                np.ix_(args.nw_roi_idxs[nw1], args.nw_roi_idxs[nw2]) + (idx_bs,)
            ]
            wts.append(np.sum(vals) / vals.size)
        return np.array(wts)

    def do_single_step(args, nw_weights, iscs, idx_bs):
        for idx1, nw1 in enumerate(args.major_tick_labels):
            for idx2, nw2 in enumerate(args.major_tick_labels):
                if idx1 > idx2: continue
                nw_weights[idx1, idx2, :, idx_bs] = get_ts(
                    args, iscs, 
                    nw1, nw2, 
                    idx_bs
                )
        return nw_weights
    
    nw_weights = np.zeros(
        shape=(
            len(args.major_tick_labels), # rois
            len(args.major_tick_labels), # rois
            len(iscs.keys()), # time
            args.n_bootstraps+1 # num_bootstraps
        )
    )
    for idx_bs in np.arange(args.n_bootstraps+1):
        nw_weights = do_single_step(args, nw_weights, iscs, idx_bs)

    return nw_weights

def plot_emo2_near_miss_nw_weights(args, nw_weights):

    nrows, ncols = [len(args.major_tick_labels[:])] * 2
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(4*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=None, hspace=0.4
    )

    ste = np.std(nw_weights[:, :, :, 1:], axis=-1)

    for idx1, nw1 in enumerate(args.major_tick_labels[:]):
        for idx2, nw2 in enumerate(args.major_tick_labels[:]):

            if idx1 > idx2: 
                axs[idx1, idx2].remove()
                continue

            ax = axs[idx1, idx2]

            # observed
            y = nw_weights[idx1, idx2, :, 0]
            ax.plot(
                args.SEGMENT_TRS,
                y,
                color=args.plot_colors['NEAR_MISS'],
                linewidth=3,
                # label='med.',
            )

            # confidence interval
            s = ste[idx1, idx2, :]
            ax.fill_between(
                args.SEGMENT_TRS,
                (y + s),
                (y - s),
                color=args.plot_colors['NEAR_MISS'],
                alpha=0.3,
                # label='c.i.'
            )

            # near miss event
            ax.axvline(x=0, linewidth=3, color='green', label='NM')

            if idx1 == 0: ax.set_title(f"{nw2}")
            if idx2 == idx1: ax.set_ylabel(f"{nw1}", size='large')
            ax.legend()
            ax.set_xlabel('TRs')
            ax.set_xticks(args.SEGMENT_TRS)
            ax.set_xticklabels(args.SEGMENT_TRS)
            ax.grid(True)

def plot_max_nw_weights(args, nw_weights):

    nrows, ncols = [len(args.major_tick_labels[:])] * 2
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(4*ncols, 4*nrows), 
        sharex=False, 
        sharey=False, 
        dpi=120
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=None, hspace=0.4
    )

    ste = np.std(nw_weights[:, :, :, 1:], axis=-1)

    for idx1, nw1 in enumerate(args.major_tick_labels[:]):
        for idx2, nw2 in enumerate(args.major_tick_labels[:]):

            if idx1 > idx2: 
                axs[idx1, idx2].remove()
                continue

            ax = axs[idx1, idx2]

            # observed
            x = np.arange(nw_weights.shape[2])
            y = nw_weights[idx1, idx2, :, 0]
            s = ste[idx1, idx2, :]
            ax.bar(
                x,
                y,
                align='center',
                color=args.plot_colors.values(),
                linewidth=3,
                yerr=s
                # label='med.',
            )

            if idx1 == 0: ax.set_title(f"{nw2}")
            if idx2 == idx1: ax.set_ylabel(f"{nw1}", size='large')
            # ax.legend()
            ax.set_xlabel('block')
            ax.set_xticks(x)
            ax.set_xticklabels(args.plot_blocks)
            ax.grid(True)