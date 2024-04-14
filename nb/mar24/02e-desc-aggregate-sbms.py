import csv
import os
import sys
import numpy as np
import pandas as pd
import scipy as sp 
import pickle 

from scipy import sparse, stats
import glob
from tqdm import tqdm
import ants
from nipype.interfaces import afni
import graph_tool.all as gt

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)


# =================

def collect_sbm_fits(args, files):
    def get(name):
        l = [s for s in ssr if name in s]
        return l[0].split('-')[-1] if len(l) > 0 else '0'
    
    fits_df = []
    for file in tqdm(files):
        ssr = file.split('/')[-2].split('_')
        sub, ses, run = list(map(get, ['sub', 'ses', 'run']))
        
        with open(f'{file}', 'rb') as f:
            [g, L, pmode, modes, marginals, state, bs, Bs, dls] = pickle.load(f)
        
        df = pd.DataFrame({
            'sub':[int(sub[-2:])],
            'ses':[int(ses)],
            'run':[int(run)],
            'ssr':[ssr],
            'graph':[g],
            'sbm':[f'sbm-{args.dc}-{args.sbm}'],
            'evidence':[L],
            'state':[state],
            'pmode':[pmode],
            'modes':[modes],
            'marginals':[marginals],
            'bs':[bs],
            'Bs':[Bs],
            'dls':[dls],
        })
        fits_df.append(df)
        
    fits_df = pd.concat(fits_df)
    fits_df = fits_df.sort_values(
        by=['sub', 'ses', 'run']
    ).reset_index(drop=True)

    return fits_df

# =================

def collect_nested_partitions(args, fits_df):
    dfs = []
    for idx, row in fits_df.iterrows():
        sub, ses, run = row[['sub', 'ses', 'run']]
        modes = row['modes']
        M = len(row['bs'])

        for idx_mode, mode in enumerate(modes):
            # plausibility
            pls = int(args.num_samples*(mode.get_M() / M))
            for s in range(pls):
                # sample <pls> partiitions from mode
                b = mode.sample_nested_partition()

                df = pd.DataFrame({
                    'sub':[sub],
                    'ses':[ses],
                    'run':[run],
                    'mode':[idx_mode],
                    'sample':[s],
                    'b':[b],
                })
                dfs.append(df)
    dfs = pd.concat(dfs).reset_index(drop=True)

    pmode = gt.ModeClusterState(dfs['b'].to_list(), nested=True)
    gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    return dfs, pmode
    
def collect_partitions(args, fits_df):
    dfs = []
    for idx, row in tqdm(fits_df.iterrows()):
        sub, ses, run = row[['sub', 'ses', 'run']]
        modes = row['modes']
        M = len(row['bs'])

        for idx_mode, mode in enumerate(modes):
            # plausibility
            pls = int(args.num_samples*(mode.get_M() / M))
            for s in range(pls):
                # sample <pls> partiitions from mode
                b = mode.sample_partition()

                df = pd.DataFrame({
                    'sub':[sub],
                    'ses':[ses],
                    'run':[run],
                    'mode':[idx_mode],
                    'sample':[s],
                    'b':[b],
                })
                dfs.append(df)
    dfs = pd.concat(dfs).reset_index(drop=True)

    pmode = gt.ModeClusterState(dfs['b'].to_list())
    gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    return dfs, pmode

def he_to_v(args, g, state, he_b):
    # half-edge partition to vertex partition
    v_b = np.zeros((g.num_vertices()), dtype=np.int32)
    for v, hes in zip(g.iter_vertices(), state.half_edges):
        try:
            v_b[v] = stats.mode(he_b[hes])[0][0]
        except: 
            v_b[v] = 0
    return v_b

def collect_overlapping_partitions(args, fits_df):
    dfs = []
    for idx, row in fits_df.iterrows():
        sub, ses, run = row[['sub', 'ses', 'run']]
        g, state = row[['graph', 'state']]
        modes = row['modes']
        M = len(row['bs'])

        for idx_mode, mode in enumerate(modes):
            # plausibility
            pls = int(args.num_samples*(mode.get_M() / M))
            for s in range(pls):
                # sample <pls> partiitions from mode
                he_b = mode.sample_partition()
                v_b = he_to_v(args, g, state, he_b) 

                df = pd.DataFrame({
                    'sub':[sub],
                    'ses':[ses],
                    'run':[run],
                    'mode':[idx_mode],
                    'sample':[s],
                    'he_b':[he_b],
                    'v_b':[v_b],
                })
                dfs.append(df)
    dfs = pd.concat(dfs).reset_index(drop=True)
    
    pmode = gt.ModeClusterState(dfs['v_b'].to_list())
    gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    return dfs, pmode

# =================

def nested_partitions(b, state):
    state = state.copy(bs=b)
    bs = []
    for l, bl in enumerate(b):
        bl_ = np.array(state.project_level(l).get_state().a)
        bs.append(bl_)
        if len(np.unique(bl_)) == 1: break
    return bs
    
def get_nested_mode_partitions(args, M, pmode, g, state):
    mode_df = []
    for idx, mode in enumerate(pmode.get_modes()):
        b = nested_partitions(mode.get_max_nested(), state)
        df = pd.DataFrame({
            'mode':[mode],
            'w':[mode.get_M()/M],
            'sigma':[mode.posterior_cdev()],
            'partition':[b],
            'marginal':[mode.get_marginal(g)],
        })
        mode_df.append(df)
    mode_df = pd.concat(mode_df).reset_index(drop=True)
    return mode_df

def get_mode_partitions(args, M, pmode, g, state):
    mode_df = []
    for idx, mode in enumerate(pmode.get_modes()):
        b = mode.get_max(g)
        b_c, sigma_c = gt.partition_overlap_center(list(mode.get_partitions().values()))
        df = pd.DataFrame({
            'mode':[mode],
            'w':[mode.get_M()/M],
            'sigma':[mode.posterior_cdev()],
            'partition':[b],
            'center_partition':[b_c],
            'center_sigma':[sigma_c],
            'marginal':[mode.get_marginal(g)],
        })
        mode_df.append(df)
    mode_df = pd.concat(mode_df).reset_index(drop=True)
    return mode_df

def rescale(X):
    X /= np.expand_dims(np.sum(X, axis=-1), axis=-1)
    X = np.nan_to_num(X)
    X = np.round(X, decimals=3)
    return X

def vertex_overlapping_partition(row, g, state):
    he_b = row['he_b']
    v_b = np.zeros((g.num_vertices(), np.max(np.unique(he_b))+1))
    for v, hes in zip(g.iter_vertices(), state.half_edges):
        blocks, counts = np.unique(he_b[hes], return_counts=True)
        v_b[v, blocks] = counts
    v_b = rescale(v_b)
    return v_b

def get_overlapping_mode_partitions(args, M, pmode, ):
    mode_df = []
    for idx_mode, mode in enumerate(pmode.get_modes()):
        # find a partition closest to all partitions in the mode
        b_hat, sigma = gt.partition_overlap_center(list(mode.get_partitions().values()))

        # find partition(s) from the collection closest to `b_hat`
        po = lambda x: gt.partition_overlap(x, b_hat)
        dists = dfs['v_b'].apply(po).to_numpy()
        row = dfs.iloc[np.where(dists == np.max(dists))[0]].iloc[0]

        # corresponding graph for the above partition from the collection
        fits_row = fits_df[fits_df['sub'] == row['sub']][fits_df['ses'] == row['ses']]
        g, state = fits_row[['graph', 'state']].iloc[0]

        v_b = vertex_overlapping_partition(row, g, state)

        df = pd.DataFrame({
            'mode':[mode],
            'w':[mode.get_M() / M],
            'sigma':[sigma],
            'partition':[v_b],
            'marginal':[None],
        })
        mode_df.append(df)
    mode_df = pd.concat(mode_df).reset_index(drop=True)
    return mode_df

# =================

def setup_args():
    class ARGS():
        pass

    args = ARGS()

    args.folder = sys.argv[1]
    args.SEED = int(sys.argv[2])
    
    return args

def main():
    args = setup_args()
    gt.seed_rng(args.SEED)
    np.random.seed(args.SEED)

    fs = args.folder.split('/')
    fs_new = fs[:-2] + [f'niis/group']
    out_folder = '/'.join(fs_new)
    print(out_folder)
    return None


# =========================

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()