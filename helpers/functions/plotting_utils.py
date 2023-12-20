import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams["errorbar.capsize"] = 0.5

import cmasher as cmr  # CITE ITS PAPER IN YOUR MANUSCRIPT


def create_matrix_ticks(args):
    args.ticks = args.num_rois
    minor_ticks = np.cumsum(args.ticks)
    args.major_ticks = minor_ticks - args.ticks // 2 - 1
    args.minor_ticks = minor_ticks[:-1]
    args.major_tick_labels = args.group_label
    return args

def set_matrix_ticks(args, ax) -> None:
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
        ax.grid(which='minor', color='k', linestyle='-', linewidth=1.5)
    except:
        pass
    
def display_network(args, network) -> None:
    nrows, ncols = 1, 1
    figsize = (5*ncols, 4*nrows)
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols,
        figsize=figsize,
        sharex=True, 
        sharey=True, 
        dpi=120,
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=None, hspace=0.5
    )

    ax = axs
    im = ax.imshow(network, cmap=args.cmap, vmin=-1, vmax=1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"network")
    set_matrix_ticks(args, ax)

    return None


def display_networks(args, networks) -> None:
    nrows, ncols = 1, len(networks)
    figsize = (5*ncols, 4*nrows)
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols,
        figsize=figsize,
        sharex=True, 
        sharey=True, 
        dpi=120,
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=None, hspace=0.5
    )

    for idx, network in enumerate(networks):
        ax = axs[idx] if ncols > 1 else axs
        im = ax.imshow(network, cmap=args.cmap, vmin=-1, vmax=1)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f"{idx+1:02}: {args.networks[idx][0].name}")
        set_matrix_ticks(args, ax)

    return None

def get_min_max(fc):
    fc = fc.flatten()
    vmin, vmax = np.min(fc), np.max(fc)
    return -max(vmin, vmax), max(vmin, vmax)

def display_fcs(args, networks, fcs) -> None:
    nrows, ncols = args.num_subjs, 1+args.num_sigmas
    figsize = (5*ncols, 4*nrows)
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols,
        figsize=figsize,
        sharex=False, 
        sharey=False, 
        dpi=120,
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=0.5, hspace=0.5
    )

    for idx_subj, network in enumerate(networks):
        ax = axs[idx_subj, 0] if nrows > 1 else axs[idx_subj]
        im = ax.imshow(network, cmap=args.cmap, vmin=-1, vmax=1)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_ylabel(f"subj{idx_subj+1:02}",fontsize='large')
        ax.set_title(f"{args.networks[idx_subj][0].name}")
        set_matrix_ticks(args, ax)

    for (idx_subj, idx_sigma) in tqdm(
        list(product(range(args.num_subjs), range(args.num_sigmas)))
    ):
        ax = axs[idx_subj, idx_sigma+1] if nrows > 1 else axs[idx_sigma+1]
        vmin, vmax = get_min_max(fcs[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"])
        im = ax.imshow(
            fcs[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"], 
            cmap=args.cmap, vmin=vmin, vmax=vmax
        )
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_ylabel(f"subj{idx_subj+1:02}",fontsize='large')
        ax.set_title(f"sig. {args.sigmas[idx_sigma]}")
        set_matrix_ticks(args, ax)

    return None

def plot_roi_time_series(args, out_dicts, roi_labels=None) -> None:
    if any('run' in k for k in list(out_dicts.keys())):
        times = np.stack(
            [out_dicts[f"run{idx_run:02}"]['t'] for idx_run in range(args.num_runs)],
            axis=0,
        )
        xs = np.stack(
            [out_dicts[f"run{idx_run:02}"]['x'] for idx_run in range(args.num_runs)],
            axis=0,
        )
        time = np.mean(times, axis=0)
        data_mean = np.mean(xs, axis=0)
        data_std = 1.00 * np.std(xs, axis=0) #/ np.sqrt(xs.shape[0])
    elif any('x' in k for k in list(out_dicts.keys())):
        time = out_dicts['t']
        data_mean = out_dicts['x']
        data_std = np.zeros_like(data_mean)        

    # plot the time series of all rois.
    # %matplotlib inline
    if args.subplot_layout == 'row-col':
        nrows, ncols = int(np.ceil(args.num_rois / 5)), 5
        figsize = (5*ncols, 4*nrows)
    elif args.subplot_layout == 'row':
        nrows, ncols = args.num_rois, 1
        figsize = (10*ncols, 4*nrows)
    fig, axs = plt.subplots(
        nrows=nrows, 
        ncols=ncols,
        figsize=figsize,
        sharex=False, 
        sharey=True, 
        dpi=120,
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=None, hspace=0.5
    )

    for idx_roi, roi in enumerate(np.arange(data_mean.shape[-1])):
        if args.subplot_layout == 'row-col':
            ax = axs[idx_roi // ncols, idx_roi % ncols] if nrows > 1 else axs[idx_roi % ncols]
        elif args.subplot_layout == 'row':
            ax = axs[idx_roi]
        
        if roi_labels is not None:
            ax.set_title(f"{roi_labels[roi]}")
        else:
            ax.set_title(f"roi {idx_roi+1:02}")

        ax.plot(
            time,
            data_mean[:, idx_roi],
            color='cornflowerblue',
            linewidth=3,
        )

        y1 = data_mean[:, idx_roi] - data_std[:, idx_roi]
        y2 = data_mean[:, idx_roi] + data_std[:, idx_roi]
        ax.fill_between(
            x=time, 
            y1=y1,
            y2=y2,
            color='cornflowerblue',
            alpha=0.5,
        )

        ax.plot(
            time,
            np.zeros_like(data_mean[:, idx_roi]),
            color='black',
            linewidth=1.5,
            linestyle='-.',
            alpha=0.5
        )

        ax.set_xlabel(f"time (ms)")
        ax.set_ylabel(f"activity")

        ax.grid(True)

    # fig.show()

    return None

def plot_fc_dists_hists(args, dists,):
    dists_df = pd.DataFrame(data=None, columns=['subj', 'sigma', 'run', 'dist'])
    for (idx_subj, idx_sigma) in tqdm(
        list(product(range(args.num_subjs), range(args.num_sigmas)))
    ):
        df = pd.DataFrame({
            'subj': [idx_subj] * args.num_runs,
            'sigma': [args.sigmas[idx_sigma]] * args.num_runs,
            'run': list(range(args.num_runs)),
            'dist': dists[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"],
        })
        dists_df = pd.concat([dists_df, df], ignore_index=True)
    
    fig, axs = plt.subplots(
        nrows=1, 
        ncols=1,
        figsize=(15, 5),
        sharex=True, 
        sharey=True, 
        dpi=120,
    )
    ax = axs
    sns.histplot(
        data=dists_df, 
        x='dist',
        hue='sigma',
        kde=True,
        stat='density',
        log_scale=args.log_scale,
        ax=ax,
        palette=mpl.colormaps['Set1'],
        line_kws={'linewidth':3},
    )
    ax.set_title(f"{args.dist_name} distance")
    ax.grid(True)

    return None