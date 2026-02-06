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

def posterior_modes(args, bs):
    pmode = gt.ModeClusterState(bs, nested=args.nested)
    gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    return pmode

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

def fit_sbm(args, g, state):
    state, bs, Bs, Bes, dls = mcmc_eq(args, g, state)
    args.nested = False
    pmode = posterior_modes(args, bs[-args.num_draws:]) # after chain equilibrates
    modes = pmode.get_modes()
    L = total_evidence(pmode, bs, dls)
    return state, bs, Bes, Bs, dls, pmode, modes, L

def fit_nested_sbm(args, g, state):
    state, bs, Bs, Bes, dls = nested_mcmc_eq(args, g, state)
    args.nested = True
    pmode = posterior_modes(args, bs[-args.num_draws:]) # after chain equilibrates
    modes = pmode.get_modes()
    L = nested_total_evidence(pmode, bs, dls)
    return state, bs, Bes, Bs, dls, pmode, modes, L

def sbm_name(args):
    dc = f'dc' if args.dc else f'nd'
    dc = f'' if args.sbm in ['a'] else dc
    file = f'sbm-{dc}-{args.sbm}'
    return file

def setup_args():
    class ARGS():
        pass
    args = ARGS()
    
    args.all_graphs_file = sys.argv[1]
    args.graph_file = int(sys.argv[2]) # file of graph, line number (index) in all_graphs.txt
    args.sbm = sys.argv[3] # a d o h
    args.dc = sys.argv[4] == 'True' # degree corrected?
    args.wait = int(sys.argv[5]) # 12,000
    args.force_niter = int(sys.argv[6]) # 25,000
    args.B = int(sys.argv[7]) # 1
    args.SEED = int(sys.argv[8]) # random seed
    
    return args

def main():
    args = setup_args()
    print(args.B)
    
    with open(f'{args.all_graphs_file}', 'r') as f:
        all_graphs = f.readlines()
        for idx, file in enumerate(all_graphs):
            all_graphs[idx] = file[:-1] if file[-1] == '\n' else file
    
    args.graph_file = all_graphs[args.graph_file]
    g = gt.load_graph(args.graph_file)
    
    fs = args.graph_file.split('/')
    ROI_RESULTS_path = '/'.join(fs[:-2])
    SBM_path = f'{ROI_RESULTS_path}/sbms/{"_".join(fs[-1].split("_")[:-1])}'
    os.system(f'mkdir -p {SBM_path}')
    
    state_df = pd.DataFrame(columns=['a', 'd', 'o', 'h'],)
    state_df.loc['state'] = [
        gt.PPBlockState(
            g,
        ), 
        gt.LayeredBlockState(
            g,
            ec=g.ep.weight,
            B=args.B,
            layers=True,
            deg_corr=args.dc,
            overlap=False,
        ), 
        gt.LayeredBlockState(
            g, 
            ec=g.ep.weight,
            B=args.B,
            layers=True,
            deg_corr=args.dc,
            overlap=True,
        ), 
        gt.NestedBlockState(
            g,
            base_type=gt.LayeredBlockState,
            state_args=dict(
                ec=g.ep.weight,
                B=args.B,
                layers=True, 
                deg_corr=args.dc, 
                overlap=False,
            )    
        ),
    ]
    
    state = state_df[args.sbm]['state']
    
    args.niter = 10
    args.num_draws = int((4/5) * args.force_niter)
    if args.sbm in ['a', 'd']:
        state, bs, Bes, Bs, dls, pmode, modes, L = fit_sbm(args, g, state)

    if args.sbm in ['h']:
        state, bs, Bes, Bs, dls, pmode, modes, L = fit_nested_sbm(args, g, state)
        
    FILE = f'{sbm_name(args)}_B-{args.B}_desc-chain.pkl'
    with open(f'{SBM_path}/{FILE}', 'wb') as f:
        pickle.dump([g, state, bs, Bs, Bes, dls, modes, L], f)
    print(f'{SBM_path}/{FILE}')
    
    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()