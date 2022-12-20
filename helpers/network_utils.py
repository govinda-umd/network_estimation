from math import floor
import os 
import sys
from os.path import join as pjoin
import numpy as np
import scipy as sp
from scipy import signal
import pandas as pd
from tqdm.notebook import tqdm
import pickle, random
import csv
import bct
from copy import deepcopy

# niimg
import nilearn
from nilearn import (image, masking)

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams["errorbar.capsize"] = 0.5

import pycircos

import cmasher as cmr #CITE ITS PAPER IN YOUR MANUSCRIPT

def consensus_partition(
    args, observed_fcs, blocks=['high_Threat', 'low_Threat']
):
    '''
    A promising approach is combining the information of the different 
    outputs into a new partition. Consensus clustering is based on this idea. 
    The goal is searching for a consensuspartition, that is 
    better fitting than the input partitions. Consensus clustering
    is a difficult combinatorial optimisation problem. 
    An alternative greedy strategy relies on the consensus matrix, 
    which is a matrix based on the co-occurrence of vertices 
    in communities of the input partitions (Fig. 22). 
    The consensus matrix is used as an input for the graph 
    clustering technique adopted, leading to a new set of 
    partitions, which produce a new consensus matrix, etc., 
    until a unique partition is finally reached, which is not 
    changed by further iterations. 
    
    The steps of the procedure are listed below. 
    The starting point is a network `G` with `n` vertices 
    and a clustering algorithm `A`.

    1. Apply `A` on `G` `n_p` times, yielding `n_p` partitions.
    2. Compute the consensus matrix `D`: `D_{ij}` 
        is the number of partitions in which vertices `i` and `j` of `G` 
        are assigned to the same community, divided by `n_p`.
    3. all entries of `D` below a chosen threshold `tau` are set to 0.
    4. Apply `A` on `D` `n_p` times, yielding `n_p` partitions.
    5. If the partitions are all equal, stop. Otherwise go back to step 2.     
    '''
    agreement_mat = {}; partition = {}; individual_partitions = {}
    # for block in tqdm(observed_fcs.keys()):
    for block in tqdm(blocks):
        # initial consensus matrix, D
        cis = np.zeros((args.num_rois, args.num_reps))
        for idx_n in np.arange(args.num_reps):
            ci, q = bct.modularity.community_louvain(
                W=observed_fcs[block],
                gamma=args.gamma,
                ci=None,
                B='modularity', #'negative_sym'
                # seed=args.SEED
            )
            cis[:, idx_n] = ci
        D = bct.clustering.agreement(cis) / args.num_reps
        agreement_mat[block] = D
        individual_partitions[block] = cis

        # consensus partition
        ciu = bct.clustering.consensus_und(
            D,
            tau=args.tau,
            gamma=args.gamma,
            reps=args.num_reps,
            seed=args.SEED
        )
        partition[block] = ciu
    
    return agreement_mat, partition, individual_partitions

def segregate_rois_into_networks(args, partition, block=None, ci=None):
    '''
    community assigment
    '''
    if (block is not None) & (ci is None):
        args.nw_names = partition[block]
    elif ci is not None:
        args.nw_names = ci
    
    '''
    plotting tick labels
    '''
    ticks = []
    for nw in np.unique(args.nw_names):
        ticks.append(np.where(args.nw_names == nw)[0].shape[0])
    args.ticks = np.array(ticks)
    print(args.ticks)

    minor_ticks = np.cumsum(args.ticks)
    args.major_ticks = minor_ticks - args.ticks // 2
    args.minor_ticks = minor_ticks[:-1]
    print(args.minor_ticks)
    print(args.major_ticks)

    args.major_tick_labels = np.unique(args.nw_names)
    print(args.major_tick_labels)

    '''
    segregate rois into networks
    '''
    args.nw_roi_idxs = {}
    for nw in np.unique(args.nw_names):
        args.nw_roi_idxs[nw] = np.where(args.nw_names == nw)[0]

    args.num_nws = len(args.nw_roi_idxs.keys())

    '''
    reorder rois
    '''
    args.roi_idxs = np.concatenate(list(args.nw_roi_idxs.values()))

    '''
    nw names to indexes
    '''
    args.nw_name_to_idx = {}
    for idx, nw in enumerate(np.unique(args.nw_names)):
        args.nw_name_to_idx[nw] = idx

    args.roi_nw_to_idx = []
    for idx in args.roi_idxs:
        args.roi_nw_to_idx += [args.nw_name_to_idx[args.nw_names[idx]]]
    args.roi_nw_to_idx = np.array(args.roi_nw_to_idx)

    return args

def plot_aba_nwwise_responses(args, X, ):
    # summary stats
    X_blocks = {}
    stats_blocks = {}
    for label, name in enumerate(X.keys()):
        X_blocks[name] = np.concatenate(X[name], axis=0)
        stats_blocks[name] = {}
        stats_blocks[name]['mean'] = np.nanmean(X_blocks[name], axis=0)
        stats_blocks[name]['ste'] = 1.96 * np.nanstd(X_blocks[name], axis=0) / np.sqrt(X_blocks[name].shape[0])
    
    # network wise plots
    for nw in args.nw_roi_idxs.keys():
        # figure inits.
        nrows, ncols = int(np.ceil(len(args.nw_roi_idxs[nw]) / 5)), 5
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
            wspace=None, hspace=0.5
        )

        fig.suptitle(f"Network {nw}")

        for idx_roi, roi in enumerate(args.nw_roi_idxs[nw]):
            if nrows > 1:
                ax = axs[idx_roi // ncols, idx_roi % ncols]
            else:
                ax = axs[idx_roi % ncols]

            ys = []
            for label, name in enumerate(stats_blocks.keys()):
                m = stats_blocks[name]['mean'][:, roi]
                s = stats_blocks[name]['ste'][:, roi]
                ax.plot(
                    m, 
                    color=args.plot_colors[name], 
                    linewidth=3,
                    # label=name
                )

                ax.fill_between(
                    np.arange(args.TRIAL_LEN),
                    (m + s),
                    (m - s),
                    color=args.plot_colors[name],
                    alpha=0.3,
                )

                ys.append(m+s)
                ys.append(m-s)

            min_y, max_y = np.min(ys), np.max(ys)
            period = 'LATE'
            ax.fill_betweenx(
                np.array([min_y, max_y]),
                x1=args.PERIOD_TRS[period][0],
                x2=args.PERIOD_TRS[period][-1],
                color='grey',
                alpha=0.3,
            )

            ax.axvline(x=8, linewidth=3, color='black', linestyle='-')
            ax.axhline(y=0, linewidth=3, color='black', linestyle='-.', alpha=0.5)

            ax.set_title(f"{roi}. {args.roi_names[roi]}")
            if idx_roi % ncols == 0: ax.set_ylabel('activation')
            ax.set_xlabel('TRs')
            # ax.legend()
            ax.grid(True)

def store_roi_vec_as_niimg(args, roi_data, mask):
    '''
    storing in nifti file, 
    for viewing in afni
    '''
    # create an empty stat img
    stat_img_all_rois = image.new_img_like(ref_niimg=mask, 
                                           data=np.zeros_like(mask.get_fdata()[..., None], 
                                                              dtype=np.float32), 
                                           copy_header=True)
    
    # unmask roi value on all voxels of the roi
    # for idx_roi in tqdm(np.arange(roi_data.shape[-1])):
    for idx_roi, roi in tqdm(enumerate(args.keep_rois_idxs)):
        mask_roi = image.math_img(f"img=={roi+1}", img=mask)
        num_voxels = np.where(mask_roi.get_fdata())[0].shape[0]
        vox_data = roi_data[:, idx_roi][:, None] @ np.ones(shape=(num_voxels,))[None, :] # time x voxels
        stat_img = masking.unmask(vox_data, mask_img=mask_roi)
        stat_img_all_rois = image.math_img(f"img_all+img_roi", 
                                           img_all=stat_img_all_rois, 
                                           img_roi=stat_img)
        
    return stat_img_all_rois

def circos_plot(args, W, ):
    def show_edge(x,):
        # theta = 1 - np.exp(x/100)
        theta = np.sqrt((x / 100))
        return np.random.rand() < theta
    
    Garc    = pycircos.Garc
    Gcircle = pycircos.Gcircle

    D_src = np.sum(W, axis=1)
    D_dest = np.sum(W, axis=0)

    circle = Gcircle(figsize=(11, 11))
    for idx_roi, roi in enumerate(args.roi_idxs):
        name = args.roi_names[roi]
        arc = Garc(
            arc_id=name,
            size=D_src[idx_roi]+D_dest[idx_roi],
            interspace=3,
            raxis_range=[900, 1000],
            facecolor=args.nw_colors[args.roi_nw_to_idx[idx_roi]],
            labelposition=250,
            labelsize=15,
            label_visible=True,
        )
        circle.add_garc(arc)
        
    circle.set_garcs()

    srcs = {roi_name:0 for roi_name in args.roi_names[args.roi_idxs]}
    dests = {
        roi_name:D_src[idx_roi] 
        for idx_roi, roi_name in enumerate(args.roi_names[args.roi_idxs])
    }
    # srcs, dests
    for idx_roi1, roi1 in enumerate(args.roi_idxs):
        for idx_roi2, roi2 in enumerate(args.roi_idxs):
            if W[idx_roi1, idx_roi2] == 0: continue
            if not show_edge(W[idx_roi1, idx_roi2]): continue
            # print(roi1, roi2, W[idx_roi1, idx_roi2])
            name1 = args.roi_names[roi1]
            nw1 = args.roi_nw_to_idx[idx_roi1]
            source = (
                name1, 
                srcs[name1], 
                srcs[name1]+W[idx_roi1, idx_roi2], 
                850
            )

            name2 = args.roi_names[roi2]
            nw2 = args.roi_nw_to_idx[idx_roi2]
            destination = (
                name2, 
                dests[name2], 
                dests[name2]+W[idx_roi1, idx_roi2], 
                850
            )

            if nw1 == nw2:
                facecolor = circle._garc_dict[name1].facecolor # src color
            else:
                facecolor = 'darkgrey' #'lightgrey' #'gainsboro' #'linen' #'oldlace' #'azure' #'whitesmoke'

            circle.chord_plot(
                source, 
                destination, 
                facecolor=facecolor,
                linewidth=0.0,
                alpha=0.5*(1 + W[idx_roi1, idx_roi2]/100),
            )

            # circle.barplot(
            #     name1,
            #     data=[0],
            #     positions=[srcs[name1]],
            #     width=[W[idx_roi1, idx_roi2]],
            #     raxis_range=[850,870],
            #     facecolor=[circle._garc_dict[name2].facecolor], #dest color
            #     spine=True
            # )

            srcs[name1] += W[idx_roi1, idx_roi2]
            dests[name2] += W[idx_roi1, idx_roi2]
            # print(source, destination)
    
    circle.ax.text(
        0, 
        0, 
        args.fig_title,
        rotation=0,
        ha='center',
        va='center',
        fontsize=25
    )

    return circle