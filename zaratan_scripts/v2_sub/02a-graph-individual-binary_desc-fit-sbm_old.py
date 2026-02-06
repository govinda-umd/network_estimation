import csv
import os
import sys
import numpy as np
import pandas as pd
import scipy as sp 
import dill as pickle 
# import pickle

from scipy import sparse, stats
from scipy.special import gammaln
import glob
from tqdm import tqdm

import graph_tool.all as gt

from multiprocessing import Process, Queue
from functools import wraps

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)

def with_timeout(seconds=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def worker(queue):
                result = func(*args, **kwargs)
                queue.put(result)

            queue = Queue()
            process = Process(target=worker, args=(queue,))
            process.start()
            process.join(seconds)

            if process.is_alive():
                process.terminate()
                return None
            
            return queue.get()
        return wrapper
    return decorator

@with_timeout(5)
def mcmc_eq(args, g, state):
    bs = [] # partitions
    Bs = np.zeros(g.num_vertices() + 1) # number of blocks
    Bes = [] # number of effective blocks
    dls = [] # description length history
    def collect_partitions(s):
        bs.append(s.b.a.copy())
        # B = s.get_nonempty_B()
        # Bs[B] += 1
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

def run_mcmc_eq(args, g, state):
    bs, Bs, Bes, dls = [], [], [], []
    while len(Bes) < args.num_iters:
        res = mcmc_eq(args, g, state)
        if res is not None:
            state, bs_, Bs_, Bes_, dls_ = res
            bs += [bs_] 
            Bs += [Bs_]
            Bes += [Bes_]
            dls += [dls_]
    bs = sum(bs, [])
    # Bs = sum(Bs, [])
    Bes = sum(Bes, [])
    dls = sum(dls, [])
    return state, bs, Bs, Bes, dls

@with_timeout(5)
def nested_mcmc_eq(args, g, state):
    bs = []
    Bs = [np.zeros(g.num_vertices() + 1) for s in state.get_levels()]
    Bes = [[] for s in state.get_levels()]
    dls = []
    def collect_partitions(s):
        bs.append(s.get_bs())
        for l, sl in enumerate(s.get_levels()):
            # B = sl.get_nonempty_B()
            # Bs[l][B] += 1
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

def run_nested_mcmc_eq(args, g, state):
    bs, Bs, Bes, dls = [], [], [], []
    while len(Bes) < args.num_iters:
        res = nested_mcmc_eq(args, g, state)
        if res is not None:
            state, bs_, Bs_, Bes_, dls_ = res
            bs += [bs_] 
            Bs += [Bs_]
            Bes += [Bes_]
            dls += [dls_]
    bs = sum(bs, [])
    # Bs = sum(Bs, [])
    Bes = sum(Bes, [])
    dls = sum(dls, [])
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
    state, bs, Bs, Bes, dls = run_mcmc_eq(args, g, state)
    args.nested = False
    pmode = posterior_modes(args, bs[-args.num_draws:]) # after chain equilibrates
    modes = pmode.get_modes()
    L = total_evidence(pmode, bs, dls)
    return state, bs, Bes, Bs, dls, pmode, modes, L

def fit_nested_sbm(args, g, state):
    state, bs, Bs, Bes, dls = run_nested_mcmc_eq(args, g, state)
    args.nested = True
    pmode = posterior_modes(args, bs[-args.num_draws:]) # after chain equilibrates
    modes = pmode.get_modes()
    L = nested_total_evidence(pmode, bs, dls)
    return state, bs, Bes, Bs, dls, pmode, modes, L

def load_graph(args, ):
    with open(f'{args.all_graphs_file}', 'r') as f:
        all_graphs = f.readlines()
        for idx, file in enumerate(all_graphs):
            all_graphs[idx] = file[:-1] if file[-1] == '\n' else file
    
    args.graph_file = all_graphs[args.graph_file]
    g = gt.load_graph(args.graph_file)
    return g

def create_state(args, ):
    state_df = pd.DataFrame(columns=['a', 'd', 'o', 'h', 'm'],)
    state_df.loc['state'] = [
        gt.PPBlockState, gt.BlockState, 
        gt.OverlapBlockState, gt.NestedBlockState,
        gt.ModularityState,
    ]
    state_df.loc['state_args'] = [
        dict(), dict(deg_corr=args.dc, B=args.B), 
        dict(deg_corr=args.dc, B=args.B), dict(deg_corr=args.dc, B=args.B),
        dict(),
    ]
    state, state_args = state_df[args.sbm]
    return state, state_args

def sbm_name(args):
    dc = f'dc' if args.dc else f'nd'
    dc = f'' if args.sbm in ['a', 'm'] else dc
    file = f'sbm-{dc}-{args.sbm}'
    return file

def save_fit(args, SBM_path, state, bs, Bs, Bes, dls, pmode, modes, L):
    with open(f'{SBM_path}/desc-state.pkl', 'wb') as f:
        pickle.dump([state], f)
    
    with open(f'{SBM_path}/desc-partitions.pkl', 'wb') as f:
        pickle.dump([bs, Bs, Bes, dls], f)

    with open(f'{SBM_path}/desc-pmode.pkl', 'wb') as f:
        pickle.dump([pmode], f)

    with open(f'{SBM_path}/desc-partition-modes.pkl', 'wb') as f:
        pickle.dump([modes], f)

    with open(f'{SBM_path}/desc-evidence.pkl', 'wb') as f:
        pickle.dump([L], f)
    return None

def setup_args():
    class ARGS():
        pass
    args = ARGS()
    
    args.all_graphs_file = sys.argv[1]
    args.graph_file = int(sys.argv[2]) # file of graph, line number (index) in all_graphs.txt
    args.sbm = sys.argv[3] # a d o h m
    args.dc = sys.argv[4] == 'True' # degree corrected?
    args.wait = int(sys.argv[5]) # 24,000
    args.total_samples = int(sys.argv[6]) # 40,000
    args.B = int(sys.argv[7]) # 1
    args.SEED = int(sys.argv[8]) # random seed

    # we will run mcmc_eq this many times
    args.force_niter = 10
    args.num_iters = args.total_samples // args.force_niter
    
    return args

def main():
    args = setup_args()
    print(args.B)
    
    g = load_graph(args)
    print(args.graph_file)
    
    fs = args.graph_file.split('/')
    ROI_RESULTS_path = '/'.join(fs[:-2])
    SBM_path = (
        f'{ROI_RESULTS_path}/model-fits'
        f'/{"_".join(fs[-1].split("_")[:-1])}' 
        f'/{sbm_name(args)}/B-{args.B}'
    )
    os.system(f'mkdir -p {SBM_path}')
    
    state, state_args = create_state(args)
    args.num_draws = int((1/2) * args.total_samples) # remove burn-in period
    args.niter = 10
    print(args.graph_file)
    if args.sbm in ['a', 'o', 'm']:
        state = gt.minimize_blockmodel_dl(g, state=state, state_args=state_args)
        # state = state(g, **state_args)
        state, bs, Bes, Bs, dls, pmode, modes, L = fit_sbm(args, g, state)

    if args.sbm in ['d']:
        state = state(g, **state_args)
        state, bs, Bes, Bs, dls, pmode, modes, L = fit_sbm(args, g, state)

    if args.sbm in ['h']:
        state = state(g, **state_args)
        state, bs, Bes, Bs, dls, pmode, modes, L = fit_nested_sbm(args, g, state)
        
    save_fit(args, SBM_path, state, bs, Bs, Bes, dls, pmode, modes, L)
    print('saved all files')

    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()