import csv
import os
import sys
import numpy as np
import pandas as pd
import scipy as sp 
import dill as pickle 
from os.path import join as pjoin
from itertools import product
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
import subprocess
from scipy import sparse, stats
from multiprocessing import Pool
import glob
import random

import arviz as az

from itertools import product, combinations
import multiprocessing as mp
from functools import partial

# networks
import graph_tool.all as gt

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)

def set_seed(args):
        gt.seed_rng(args.SEED)
        np.random.seed(args.SEED)

def setup_args():
    class ARGS():
        pass

    args = ARGS()

    args.type = sys.argv[1] #'spatial'
    args.roi_size = sys.argv[2] #225
    args.maintain_symmetry = sys.argv[3] #True
    args.brain_div = sys.argv[4] #'whl'
    args.num_rois = sys.argv[5] #162

    args.GRAPH_DEF = sys.argv[6] #f'constructed'
    args.GRAPH_METHOD = sys.argv[7] #f'pearson-corr'
    args.THRESHOLDING = sys.argv[8] #f'positive'
    args.EDGE_DEF = sys.argv[9] #f'binary'
    args.EDGE_DENSITY = int(sys.argv[10]) #10
    args.LAYER_DEF = sys.argv[11] #f'individual'
    args.DATA_UNIT = sys.argv[12] #f'ses'

    args.dc = sys.argv[13] == 'True' # True/False
    args.sbm = sys.argv[14] # 'a', 'd', 'h'

    args.SEED = int(sys.argv[15]) #100

    set_seed(args)

    return args

def collect_indiv_dfs(args, indiv_files):
    # individual estimates, per animal
    indiv_dfs = []
    for indiv_file in tqdm(indiv_files):
        with open(indiv_file, 'rb') as f:
            indiv_df = pickle.load(f)
        sub = indiv_file.split('/')[-3].split('-')[-1]
        indiv_df['sub'] = [sub]*len(indiv_df)
        cols = list(indiv_df.columns)
        reordered_cols = [cols[-1]] + cols[:-1]
        indiv_df = indiv_df.reindex(columns=reordered_cols)
        indiv_dfs += [indiv_df]
        # break
    indiv_dfs = pd.concat(indiv_dfs).reset_index(drop=True)
    return indiv_dfs

def nested_partitions(g, b):
    b = gt.nested_partition_clear_null(b)
    state = gt.NestedBlockState(g, bs=[g.new_vp("int", vals=b[0])] + b[1:])
    state = state.copy(bs=b)
    bs = []
    for l, bl in enumerate(b):
        bl_ = np.array(state.project_level(l).get_state().a)
        bs.append(bl_)
        if len(np.unique(bl_)) == 1: break
    return bs

def project_partitions_on_graph(args, g, mode, max_level=-1):
    proj_bs = []
    for bs in tqdm(list(mode.get_nested_partitions().values())):
        proj_bs += [nested_partitions(g, bs)]
    max_level = np.max([len(bs) for bs in proj_bs]) if max_level == -1 else max_level
    level_bs = [[] for _ in range(max_level)]
    for bs in proj_bs:
        for level in range(max_level):
            level_bs[level] += [bs[level] if len(bs) > level else [0]*len(bs[0])]
    return level_bs, max_level

def align_nested_mode_to_pmode(args, g, mode, pmode_level_bs, pmode_max_level):
    mode_level_bs, mode_max_level = project_partitions_on_graph(args, g, mode, max_level=pmode_max_level)
    
    gmodes = []
    for level in tqdm(range(pmode_max_level)):
        gmode = gt.PartitionModeState(mode_level_bs[level], relabel=False, nested=False, converge=True)
        pmode_level = gt.PartitionModeState(pmode_level_bs[level], relabel=False, nested=False, converge=False)
        gmode.align_mode(pmode_level)
        gmodes += [gmode]
    
    return gmodes

def get_pi_matrix(args, mrgnls):
    num_comms = np.max([len(mrgnl) for mrgnl in mrgnls])
    pi = np.zeros((len(mrgnls), num_comms))

    for idx_node, mrgnl in enumerate(mrgnls):
        mrgnl = np.array(mrgnl)
        pi[idx_node, np.where(mrgnl)[0]] = mrgnl[mrgnl > 0]

    pi = pi / np.expand_dims(pi.sum(axis=-1), axis=-1)
    return pi # marginals matrix

def get_nested_marginals(args, g, level_modes):
    marginals = [list(level_mode.get_marginal(g)) for level_mode in level_modes]
    pis = {}
    for level, mrgnls in enumerate(marginals):
        pis[level] = get_pi_matrix(args, mrgnls)
    return pis

def collect_marginals_single_mode(args, row, pi):
    df = pd.DataFrame()
    df['sub'] = [row['sub']]
    df['mode_id'] = [row['mode_id']]
    df['pi'] = [pi]
    df['omega'] = [row['omega']]
    df['sigma'] = [row['sigma']]
    df['ratio'] = [row['ratio']]
    return df

def collect_nested_marginals_single_mode(args, row, pis):
    dfs = []
    for level, pi in pis.items():
        df = pd.DataFrame()
        df['sub'] = [row['sub']]
        df['mode_id'] = [row['mode_id']]
        df['level'] = [level]
        df['pi'] = [pi]
        df['omega'] = [row['omega']]
        df['sigma'] = [row['sigma']]
        df['ratio'] = [row['ratio']]
        dfs += [df]

    dfs = pd.concat(dfs).reset_index(drop=True)
    return dfs

def post_align_modes(args, indiv_dfs, g):
    pmode = gt.PartitionModeState(indiv_dfs['b_hat'].to_list(), nested=args.nested, converge=True)
    pmode_level_bs, pmode_max_level = project_partitions_on_graph(args, g, pmode)

    if args.sbm in ['h']:
        indiv_marginals_dfs = []
        for idx, row in indiv_dfs.iterrows():
            mode = row['mode']
            aligned_level_modes = align_nested_mode_to_pmode(args, g, mode, pmode_level_bs, pmode_max_level)
            pis = get_nested_marginals(args, g, aligned_level_modes)
            marginal_df = collect_nested_marginals_single_mode(args, row, pis)
            indiv_marginals_dfs += [marginal_df]
            # break
        indiv_marginals_dfs = pd.concat(indiv_marginals_dfs).reset_index(drop=True)

    if args.sbm in ['a', 'd']:
        indiv_marginals_dfs = []
        for idx, row in tqdm(indiv_dfs.iterrows()):
            mode = row['mode']
            mode.align_mode(pmode) # align to the group 
            mrgnls = list(mode.get_marginal(g))
            pi = get_pi_matrix(args, mrgnls)
            df = collect_marginals_single_mode(args, row, pi)
            indiv_marginals_dfs += [df]
            # break
        indiv_marginals_dfs = pd.concat(indiv_marginals_dfs).reset_index(drop=True)
    
    return indiv_marginals_dfs

def main():
    args = setup_args()

    PARC_DESC = (
        f'type-{args.type}'
        f'_size-{args.roi_size}'
        f'_symm-{args.maintain_symmetry}'
        f'_braindiv-{args.brain_div}'
        f'_nrois-{args.num_rois}'
    )

    BASE_path = f'{os.environ["HOME"]}/mouse_dataset/roi_results_v2'
    ROI_path = f'{BASE_path}/{PARC_DESC}'
    TS_path = f'{ROI_path}/runwise_timeseries'
    ROI_RESULTS_path = (
        f'{ROI_path}'
        f'/graph-{args.GRAPH_DEF}/method-{args.GRAPH_METHOD}'
        f'/threshold-{args.THRESHOLDING}/edge-{args.EDGE_DEF}/density-{args.EDGE_DENSITY}'
        f'/layer-{args.LAYER_DEF}/unit-{args.DATA_UNIT}'
    )
    GRAPH_path = f'{ROI_RESULTS_path}/graphs'
    os.system(f'mkdir -p {GRAPH_path}')
    SBM_path = f'{ROI_RESULTS_path}/model-fits'
    os.system(f'mkdir -p {SBM_path}')
    ESTIM_path = f'{ROI_RESULTS_path}/estimates'
    os.system(f'mkdir -p {ESTIM_path}/individual')
    os.system(f'mkdir -p {ESTIM_path}/group')

    graph_file = sorted(glob.glob(f'{GRAPH_path}/*', recursive=True))[0]
    g = gt.load_graph(graph_file)

    args.nested = args.sbm == 'h'

    args.force_niter = 40000
    args.num_draws = int((1/2) * args.force_niter)

    def sbm_name(args):
        dc = f'dc' if args.dc else f'nd'
        dc = f'' if args.sbm in ['a'] else dc
        file = f'sbm-{dc}-{args.sbm}'
        return file

    SBM = sbm_name(args)
    print(SBM)

    indiv_files = sorted(glob.glob(f'{ESTIM_path}/individual/sub-*/partition-modes/{SBM}_desc-df.pkl', recursive=True))

    indiv_dfs = collect_indiv_dfs(args, indiv_files)

    indiv_marginals_dfs = post_align_modes(args, indiv_dfs, g)

    for sub in tqdm(indiv_marginals_dfs['sub'].unique()):
        folder = f'{ESTIM_path}/individual/sub-{sub}/partition-modes-group-aligned/{SBM}'
        os.system(f'mkdir -p {folder}')
        with open(f'{folder}/desc-marginals-df.pkl', 'wb') as f:
            pickle.dump(indiv_marginals_dfs[indiv_marginals_dfs['sub'] == sub].reset_index(drop=True), f)

    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()