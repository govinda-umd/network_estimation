import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import (norm, zscore, permutation_test)
from itertools import combinations

# ISC
from brainiak.isc import (
    isc, isfc, bootstrap_isc, compute_summary_statistic, squareform_isfc, compute_correlation,
    _check_timeseries_input, _check_targets_input, _threshold_nans
)
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import squareform

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams["errorbar.capsize"] = 0.5

import cmasher as cmr #CITE ITS PAPER IN YOUR MANUSCRIPT

# DATA TIME SERIES
# -----------------
def get_max_block_time_series(args, X, ):
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

# def get_aba_block_time_series(args, X):
#     '''
#     find minimum number of trials across subjects
#     '''
#     min_trials = []
#     for idx, _ in enumerate(args.LABEL_NAMES):
#         min_trials += [x.shape[0] for x in X[idx]]
#     min_trials = min(min_trials)
#     print(f"minimum number of trials = {min_trials}")

#     '''
#     time series for the late period
#     '''
#     args.TR = 1.25
#     args.LATE_PERIOD_TRS = (np.arange(-3.75, 1.25+args.TR, args.TR) // args.TR + 8.0).astype(int)
#     # because play block/trial starts at -8TR and ends at 4TR, and play period ends at 0TR.
#     ts = {}
#     for label, name in enumerate(args.LABEL_NAMES):
#         ts[f"{name}"] = []
#         for x in X[label]:
#             x = x[:, args.LATE_PERIOD_TRS, :]
#             trl, t, r = x.shape
#             x = np.reshape(x[:min_trials, ...], (min_trials*t, r))
#             ts[f"{name}"] += [zscore(x, axis=0, nan_policy='omit')]
    
#     for block in ts.keys():
#         ts[block] = np.stack(ts[block], axis=0)
    
#     return ts

def get_aba_block_time_series(args, X, trial_option='concat', subj_axis=0, period='LATE'):
    ts = {}
    for name in args.NAMES:
        block = name
        ts[block] = []
        for idx_subj, x in enumerate(X[name]):
            x = x[:, args.PERIOD_TRS[period], :] # trial, time, roi
            
            if trial_option == 'concat':
                ts[block].append(
                    np.concatenate(x, axis=0)
                )
            elif trial_option == 'mean':
                ts[block].append(
                    np.nanmean(x, axis=0)
                )
                
        # ts[block] = np.stack(ts[block], axis=subj_axis) # (subj, time, roi)

    return ts

# FC MATRICES
# -----------------
def get_fcs(args, ts, print_stats=False):
    corrs = {}; bootstraps = {}; rois = {}

    for block in ts.keys():
        # FC matrices
        corrs[block] = []

        for idx_subj in range(len(ts[block])):
            x = ts[block][idx_subj].T
            corrs[block].append(
                squareform_isfc(
                    compute_correlation(
                        np.ascontiguousarray(x), 
                        np.ascontiguousarray(x), 
                        return_nans=True
                    )
                )[0] # off-diags of fc mats
            ) 

        corrs[block] = np.stack(corrs[block], axis=0)
        
        # bootstrap hypo. testing
        observed, ci, p, distribution = bootstrap_isc(
            corrs[block],
            pairwise=False,
            summary_statistic='median',
            n_bootstraps=args.n_bootstraps,
            ci_percentile=95,
            side='two-sided',
            random_state=args.SEED
        )
        bootstraps[block] = observed, ci, p, distribution

        # surviving roi-pairs
        rois[block] = bootstraps[block][2] < 0.05

        if print_stats == True:
            print(
                (
                    f"condition {block}: " 
                    f"{100. * np.sum(rois[block]) / len(rois[block])} %"
                    f"significant roi(-pairs)"
                )
            )
    
    return corrs, bootstraps, rois

def threshold_fc(args, fc, q=0.25):
    # thresh = np.nanquantile(fc.flatten(), q=q)
    # return fc * (fc >= thresh)
    
    fc_abs =  np.abs(fc.flatten())
    thresh = np.nanquantile(fc_abs, q=args.q)
    return fc * ((fc >= thresh) | (fc <= -thresh))

def get_squareform_matrices(args, bootstraps, rois, positive_only=False, threshold_mats=True,):
    observed_fcs = {}; observed_p_vals = {}; 
    significant_rois = {}; conf_intervals = {}
    for block in bootstraps.keys():
        # if block == 'safe_early': continue
        observed_fcs[block] = squareform(
            bootstraps[block][0], 
        )
        observed_p_vals[block] = squareform(
            bootstraps[block][2], 
        )
        significant_rois[block] = squareform(
            rois[block], 
        )
        conf_intervals[block] = (
            squareform(
                bootstraps[block][1][0],
            ),
            squareform(
                bootstraps[block][1][1],
            )
        )

        if threshold_mats:
            # observed_fcs[block] *= significant_rois[block]
            observed_fcs[block] = threshold_fc(args, observed_fcs[block], args.q)

        if positive_only:
            observed_fcs[block] *= (observed_fcs[block] > 0)
    
    return observed_fcs, observed_p_vals, significant_rois, conf_intervals

def get_bootstrap_distribution_fcs(args, observed_fcs, bootstraps):
    bootstrap_fcs = {}
    all_fcs = {}
    for block in bootstraps.keys():
        bootstrap_fcs[block] = bootstraps[block][3]

        all_fcs[block] = np.concatenate(
            [
                squareform_isfc(observed_fcs[block])[0][None, :],
                bootstrap_fcs[block]
            ],
            axis=0
        )
    return bootstrap_fcs, all_fcs

def separate_pos_neg_weights(args, all_fcs, significant_rois, threshold_mats=True):
    all_fcs_pos = {}; all_fcs_neg = {}
    all_sq_fcs_pos = {}; all_sq_fcs_neg = {}
    for block in all_fcs.keys():
        all_fcs_pos[block] = np.multiply(
            all_fcs[block] > 0,
            all_fcs[block]
        )

        all_fcs_neg[block] = -1 * np.multiply(
            all_fcs[block] < 0,
            all_fcs[block]
        )  

        all_sq_fcs_pos[block] = np.zeros((args.num_rois, args.num_rois, args.n_bootstraps+1))
        all_sq_fcs_neg[block] = np.zeros((args.num_rois, args.num_rois, args.n_bootstraps+1))
        for idx_bs in np.arange(len(all_fcs_pos[block])):
            all_sq_fcs_pos[block][:, :, idx_bs] = squareform(all_fcs_pos[block][idx_bs])

            all_sq_fcs_neg[block][:, :, idx_bs] = squareform(all_fcs_neg[block][idx_bs])
    
    return all_fcs_pos, all_fcs_neg, all_sq_fcs_pos, all_sq_fcs_neg

def get_min_max(d):
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
    return -max(-vmin, vmax), max(-vmin, vmax)

def plot_max_fcs(args, fcs, rois, cmap=cmr.iceburn): 
    vmin, vmax = get_min_max(fcs)

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
                fcs[block], #* rois[block], 
                cmap=cmap, 
                # vmin=vmin, vmax=vmax
                vmin=0.0, vmax=vmax
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

def plot_aba_fcs(args, fcs, rois):
    vmin, vmax = get_min_max(fcs)

    nrows, ncols = [len(args.LABEL_NAMES)//2]*2
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

    for idx_valence, valence in enumerate(args.VALENCE):
        for idx_level, level in enumerate(args.LEVELS):
            ax = axs[idx_valence, idx_level]

            block = f"PLAY_{level}{valence[0]}"

            im = ax.imshow(
                fcs[block], #* roi[block],
                cmap=cmr.iceburn, vmin=vmin, vmax=vmax
            )
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if idx_valence == 0: ax.set_title(f"{level}")
            if idx_level == 0: ax.set_ylabel(f"{valence}", size='large')

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

# COMPARISONS BETWEEN FC MATRICES
# --------------------------
def get_fc_comparison_stats(args, corrs, paradigm='max'):
    def statistic(obs1, obs2, axis):
        return (
            + compute_summary_statistic(obs1, summary_statistic='median', axis=axis) 
            - compute_summary_statistic(obs2, summary_statistic='median', axis=axis)
        ) 

    comp_results = {}
    if paradigm == 'max':
        blocks = list(corrs.keys())[::-1]
    else:
        blocks = list(corrs.keys())

    for (block1, block2) in tqdm(combinations(blocks, 2)):
        obs1 = corrs[block1]
        obs2 = corrs[block2]

        comp_results[(block1, block2)] = permutation_test(
            data=(obs1, obs2),
            statistic=statistic,
            permutation_type='samples',
            vectorized=True,
            n_resamples=args.n_resamples,
            batch=5,
            alternative='two-sided',
            axis=0,
            random_state=args.SEED
        )
        
    return comp_results

def get_diff_fcs(args, comp_results, threshold_mats=True):
    diff_fcs = {}; diff_pvals = {}

    for (block1, block2) in comp_results.keys():
        diff_fcs[(block1, block2)] = squareform(comp_results[(block1, block2)].statistic)

        diff_pvals[(block1, block2)] = comp_results[(block1, block2)].pvalue
        diff_pvals[(block1, block2)] = squareform(diff_pvals[(block1, block2)] < 0.05)

        if threshold_mats == True:
            diff_fcs[(block1, block2)] *= diff_pvals[(block1, block2)]
    
    return diff_fcs, diff_pvals

def plot_fc_comparisons(args, corrs, diff_isfcs, paradigm='max'):
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

    if paradigm == 'max':
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

            ax.set_yticks(args.major_ticks, args.major_tick_labels, rotation=0, va='center')
            ax.set_xticks(args.major_ticks, args.major_tick_labels, rotation=90, ha='center')

            ax.set_yticks(args.minor_ticks-0.5, minor=True)
            ax.set_xticks(args.minor_ticks-0.5, minor=True)
            ax.tick_params(
                which='major', direction='out', length=7, 
                # grid_color='white', grid_linewidth='1.5',
                labelsize=10,
            )
            ax.grid(which='minor', color='w', linestyle='-', linewidth=1.5)
    
    # fig.tight_layout()

def plot_max_fc_comparisons(args, corrs, diff_isfcs):
    plot_fc_comparisons(args, corrs, diff_isfcs, 'max')

def plot_aba_fc_comparisons(args, corrs, diff_isfcs):
    plot_fc_comparisons(args, corrs, diff_isfcs, 'aba')