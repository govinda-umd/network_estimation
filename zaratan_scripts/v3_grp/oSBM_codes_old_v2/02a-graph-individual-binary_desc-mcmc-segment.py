import os
import sys
import numpy as np
import pandas as pd
import dill as pickle 
# import pickle
import pprint
import time

from scipy import sparse, stats
from scipy.special import gammaln

import graph_tool.all as gt

from multiprocessing import Process, Queue
from functools import wraps

import argparse

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

def load_graph(args, ):
    with open(f'{args.all_graphs_file}', 'r') as f:
        all_graphs = f.readlines()
        for idx, file in enumerate(all_graphs):
            all_graphs[idx] = file[:-1] if file[-1] == '\n' else file
    
    args.graph_file = all_graphs[args.graph_file]
    g = gt.load_graph(args.graph_file)
    return g

def sbm_name(args):
    dc = f'dc' if args.dc else f'nd'
    dc = f'' if args.sbm in ['a', 'm'] else dc
    file = f'sbm-{dc}-{args.sbm}'
    return file

def save_state(args, SBM_path, state):
    with open(f'{SBM_path}/desc-state.pkl', 'wb') as f:
        pickle.dump([state], f)
    return None

def load_state(args, SBM_path):
    with open(f'{SBM_path}/desc-state.pkl', 'rb') as f:
        [state] = pickle.load(f)
    return state

def save_append_file(args, file, new_data):
    if os.path.exists(file):
        with open(file, 'rb') as f:
            [existing] = pickle.load(f)
    else:
        existing = []

    combined = existing + new_data

    with open(file, 'wb') as f:
        pickle.dump([combined], f)

    return None

def save_segment(args, SBM_path, state, bs, Bs, Bes, dls):
    save_state(args, SBM_path, state)
    save_append_file(args, f'{SBM_path}/desc-bs.pkl', bs)
    save_append_file(args, f'{SBM_path}/desc-Bes.pkl', Bes)
    save_append_file(args, f'{SBM_path}/desc-dls.pkl', dls)
    return None

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def setup_args():
    parser = argparse.ArgumentParser(description="Initial state finder for MCMC")

    parser.add_argument('--all-graphs-file', type=str, required=True,
                        help='Path to the text file containing all graph file paths')
    parser.add_argument('--subject-id', type=int, required=True,
                        help='Index of the subject in the graph file list')

    parser.add_argument('--sbm', type=str, choices=['a', 'd', 'o', 'h', 'm'], required=True,
                        help='SBM type: assortative, disjoint, overlapping, hierarchical, modularity')
    parser.add_argument('--dc', type=str2bool, default=True,
                        help='Use degree correction')
    parser.add_argument('--B', type=int, required=True,
                        help='Initial number of groups for initialization')
    parser.add_argument('--gamma', type=float, default=2.0,
                        help='Gamma parameter for modularity prior (if using ModularityState)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')

    # mcmc_eq() parameters used during diversity checking
    parser.add_argument('--wait', type=int, default=1200,
                        help='Wait time for mcmc_equilibrate()')
    parser.add_argument('--segment-len', type=int, default=500,
                        help='Force niter for mcmc_equilibrate()')
    parser.add_argument('--niter', type=int, default=10,
                        help='Number of iterations between samples to reduce auto-correlation')
    
    # segment identifier
    parser.add_argument('--segment-id', type=int, default=0,
                        help='Segment ID')

    args = parser.parse_args()

    # Derived quantity
    args.force_niter = args.segment_len + 1 
    # (:top) to keep the last value, otherwise num_samples = len - 1
    args.num_draws = int(0.5 * args.force_niter)
    args.graph_file = args.subject_id

    return args

def main():
    args = setup_args()
    pprint.pprint(vars(args))

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

    state = load_state(args, SBM_path)

    if args.sbm in ['a', 'd', 'm', 'o']:
        state, bs, Bs, Bes, dls = mcmc_eq(args, g, state)
        print(f'len of Bes: {len(Bes)}')

    if args.sbm in ['h']:
        pass

    save_segment(args, SBM_path, state, bs, Bs, Bes, dls)
    print(f'saved all files')

    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()