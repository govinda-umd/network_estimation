import csv
import os
import sys
import numpy as np
import pandas as pd
import scipy as sp 
import pickle 

from scipy import sparse, stats
from scipy.special import gammaln
import glob
from tqdm import tqdm

import graph_tool.all as gt

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)

# =================

def mcmc_eq(args, g, state):
    bs = [] # partitions
    Bs = np.zeros(g.num_vertices() + 1) # number of blocks
    Bes = [] # number of effective blocks
    dls = [] # description length history
    def collect_partitions(s):
        bs.append(s.b.a.copy())
        B = s.get_nonempty_B()
        Bs[B] += 1
        Bes.append(s.get_Be())
        dls.append(s.entropy())
        
    gt.mcmc_equilibrate(
        state, 
        wait=args.wait, 
        force_niter=args.force_niter,
        mcmc_args=dict(niter=args.niter), 
        callback=collect_partitions,
    )
    return state, bs, Bs, Bes, dls

def nested_mcmc_eq(args, g, state):
    bs = []
    Bs = [np.zeros(g.num_vertices() + 1) for s in state.get_levels()]
    Bes = [[] for s in state.get_levels()]
    dls = []
    def collect_partitions(s):
        bs.append(s.get_bs())
        for l, sl in enumerate(s.get_levels()):
            B = sl.get_nonempty_B()
            Bs[l][B] += 1
            Be = sl.get_Be()
            Bes[l].append(Be)
        dls.append(s.entropy())
        
    gt.mcmc_equilibrate(
        state, 
        wait=args.wait, 
        force_niter=args.force_niter, 
        mcmc_args=dict(niter=args.niter),
        callback=collect_partitions,
    )
    return state, bs, Bs, Bes, dls

# ==================

def rescale(X):
    X /= np.expand_dims(np.sum(X, axis=-1), axis=-1)
    X = np.nan_to_num(X)
    X = np.round(X, decimals=3)
    return X

def get_marginals(args, g, state, mode):
    # vertex marginals of the graph in the state, 
    # if overlapping state, it is graph of half-edges
    sg = state.g
    B = mode.get_B()
    print(f'{B} blocks')
    v_marginals = np.zeros((sg.num_vertices(), B))
    for idx, prob in zip(sg.iter_vertices(), mode.get_marginal(sg)):
        prob = list(prob)
        prob = prob if len(prob) < B else prob[:B]
        prob = prob + [0]*(B - len(prob))
        v_marginals[idx] = np.array(prob)

    v_marginals = rescale(v_marginals)
    
    if args.sbm in ['o']:
        # average of probs. of half-edges incident on a vertex
        marginals = np.zeros((g.num_vertices(), v_marginals.shape[-1]))
        for v, hes in zip(g.iter_vertices(), state.half_edges):
            marginals[v] = np.mean(v_marginals[hes], axis=0)
    else:
        marginals = v_marginals.copy()
    return marginals

def get_mode_marginals(args, g, state, bs):
    pmode = gt.ModeClusterState(bs=bs)
    gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))    
    modes = pmode.get_modes()
    print(f'{len(modes)} modes present')
    
    marginals = []
    for i, mode in enumerate(modes):
        marginals.append(get_marginals(args, g, state, mode))
        
    return pmode, modes, marginals

def get_nested_mode_marginals(args, g, state, bs):
    pmode = gt.ModeClusterState(bs, nested=True)
    gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    modes = pmode.get_modes()
    print(f'{len(modes)} modes present')
    
    marginals = []
    for i, mode in enumerate(modes):
        marginals.append(get_marginals(args, g, state, mode))
        
    return pmode, modes, marginals

# ==================

def total_evidence(pmode, bs, dls):
    H = pmode.posterior_entropy()
    logB = np.mean(gammaln(np.array([len(np.unique(b)) for b in bs]) + 1))
    L = -np.mean(dls) + logB + H
    return L

def nested_total_evidence(pmode, bs, dls):
    H = pmode.posterior_entropy()
    logB = np.mean([sum(gammaln(len(np.unique(bl))+1) for bl in b) for b in bs])
    L = -np.mean(dls) + logB + H
    return L

# ==================

def fit_sbm(args, g, state, state_args, mcmc_args):
    state = gt.minimize_blockmodel_dl(
        g, state=state, 
        state_args=state_args, 
        multilevel_mcmc_args=mcmc_args, 
    )
    print('heuristic')
    
    state, bs, Bs, Bes, dls = mcmc_eq(args, g, state,)
    print('mcmc')
    
    pmode, modes, marginals = get_mode_marginals(args, g, state, bs)
    print('mode clusters')

    L = total_evidence(pmode, bs, dls)
    print('evidence')
    
    return pmode, L, modes, marginals, state, bs, Bs, Bes, dls,

def fit_nested_sbm(args, g, state, state_args, mcmc_args):
    state = gt.minimize_nested_blockmodel_dl(
        g, state=state, 
        state_args=state_args, 
        multilevel_mcmc_args=mcmc_args, 
    )
    print('heuristic')
    
    state, bs, Bs, Bes, dls = nested_mcmc_eq(args, g, state,)
    print('mcmc')
    
    pmode, modes, marginals = get_nested_mode_marginals(args, g, state, bs)
    print('mode clusters')

    L = nested_total_evidence(pmode, bs, dls)
    print('evidence')
    
    return pmode, L, modes, marginals, state, bs, Bs, Bes, dls,

# ===================

def setup_args():
    class ARGS():
        pass
    args = ARGS()
    
    args.DESC = sys.argv[1] # parcellation
    args.graph_file = sys.argv[2] # file of graph
    args.sbm = sys.argv[3] # p d o h
    args.dc = sys.argv[4] == '1' # degree corrected?
    args.wait = int(sys.argv[5]) # 1000
    args.SEED = int(sys.argv[6]) # random seed
    
    return args

def main():
    args = setup_args()
    gt.seed_rng(args.SEED)
    np.random.seed(args.SEED)
    print(
        # args.DESC,
        args.graph_file, 
        args.sbm, args.dc, 
        args.wait, args.SEED
    )
    
    if args.sbm == 'a' and args.dc == False:
        return None
    
    fs = args.graph_file.split('/')
    if not fs[-5] == args.DESC: return None
    
    SBM_path = fs[:-2].copy()
    SBM_path += ['sbms']
    SBM_path += ['_'.join(fs[-1].split('_')[:-1])]
    SBM_path = '/'.join(SBM_path)
    os.system(f'mkdir -p {SBM_path}')
    
    
    g = gt.load_graph(args.graph_file)
    
    args.force_niter = 100000
    args.niter = 10
    
    state_df = pd.DataFrame(columns=['a', 'd', 'o', 'h'],)
    state_df.loc['state'] = [
        gt.PPBlockState, gt.BlockState, 
        gt.OverlapBlockState, gt.NestedBlockState,
    ]
    state_df.loc['state_args'] = [
        dict(), dict(deg_corr=args.dc), 
        dict(deg_corr=args.dc), dict(deg_corr=args.dc),
    ]
    state_df.loc['mcmc_args'] = [
        dict(), dict(), 
        dict(), dict(),
    ]
    state, state_args, mcmc_args = state_df[args.sbm]
    
    if not 'h' in args.sbm:
        print(f'{args.sbm} SBM')
        (
            pmode, L, 
            modes, marginals, 
            state, bs, Bs, Bes, dls,
        ) = fit_sbm(
            args, g, 
            state, state_args, 
            mcmc_args
        )
    else:
        print(f'{args.sbm} SBM')
        (
            pmode, L, 
            modes, marginals, 
            state, bs, Bs, Bes, dls,
        ) = fit_nested_sbm(
            args, g, 
            state, state_args, 
            mcmc_args
        )
    
    def file_name(args):
        dc = f'dc' if args.dc else f'nd'
        dc = f'' if args.sbm in ['a'] else dc
        file = f'sbm-{dc}-{args.sbm}_desc-fit.npy'
        return file
    
    sbm_file = file_name(args)
    file = f'{SBM_path}/{sbm_file}'
    with open(f'{file}', 'wb') as f:
        pickle.dump(
            [   
                g, L,
                pmode, modes, marginals, 
                state, Bes,
            ], 
            f
        )
    print(f'{file}')
    
    return None

# =========================

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
        

# COMMAND TO RUN IN BASH
# bash 02c-desc-fit-sbm.sh spatial 225 True whl 162 "a d h o" "1 0" 100 100