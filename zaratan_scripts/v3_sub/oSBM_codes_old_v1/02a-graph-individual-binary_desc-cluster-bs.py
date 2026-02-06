import os
import sys
import numpy as np
import pandas as pd
import dill as pickle 
# import pickle
import pprint

from scipy import sparse, stats
from scipy.special import gammaln

import graph_tool.all as gt

from multiprocessing import Process, Queue
from functools import wraps

import argparse

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)

def posterior_modes(args, bs):
    pmode = gt.ModeClusterState(bs, nested=args.nested)
    gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    return pmode

def total_evidence(pmode, bs, dls):
    H = pmode.posterior_entropy()
    logB = np.mean(gammaln(np.array([len(np.unique(b)) for b in bs]) + 1))
    L = -np.mean(dls) + logB + H
    return L

def load_graph(args, ):
    with open(f'{args.all_graphs_file}', 'r') as f:
        all_graphs = f.readlines()
        for idx, file in enumerate(all_graphs):
            all_graphs[idx] = file[:-1] if file[-1] == '\n' else file
    
    args.graph_file = all_graphs[args.graph_file]
    g = gt.load_graph(args.graph_file)
    return g

def load_data(args, SBM_path, data_name):
    with open(f'{SBM_path}/desc-{data_name}.pkl', 'rb') as f:
        [data] = pickle.load(f)
    return data

def sbm_name(args):
    dc = f'dc' if args.dc else f'nd'
    dc = f'' if args.sbm in ['a', 'm'] else dc
    file = f'sbm-{dc}-{args.sbm}'
    return file

def save_data(args, SBM_path, bs, Bs, Bes, dls):
    with open(f'{SBM_path}/desc-partitions.pkl', 'wb') as f:
        pickle.dump([bs, Bs, Bes, dls], f)

    os.system(f'rm -rf {SBM_path}/desc-bs.pkl')
    os.system(f'rm -rf {SBM_path}/desc-Bes.pkl')
    os.system(f'rm -rf {SBM_path}/desc-dls.pkl')

    return None

def save_modes(args, SBM_path, pmode, modes):
    with open(f'{SBM_path}/desc-pmode.pkl', 'wb') as f:
        pickle.dump([pmode], f)

    with open(f'{SBM_path}/desc-partition-modes.pkl', 'wb') as f:
        pickle.dump([modes], f)
    return None

def save_evidence(args, SBM_path, L):
    with open(f'{SBM_path}/desc-evidence.pkl', 'wb') as f:
        pickle.dump([L], f)
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

    # # mcmc_eq() parameters used during diversity checking
    # parser.add_argument('--wait', type=int, default=1200,
    #                     help='Wait time for mcmc_equilibrate()')
    # parser.add_argument('--segment-len', type=int, default=500,
    #                     help='Force niter for mcmc_equilibrate()')
    # parser.add_argument('--niter', type=int, default=10,
    #                     help='Number of iterations between samples to reduce auto-correlation')
    
    # # segment identifier
    # parser.add_argument('--segment-id', type=int, default=0,
    #                     help='Segment ID')

    args = parser.parse_args()

    # Derived quantity
    # args.force_niter = args.segment_len + 1 
    # (:top) to keep the last value, otherwise num_samples = len - 1
    # args.num_draws = int(0.5 * args.force_niter)
    args.graph_file = args.subject_id

    args.nested = True if args.sbm in ['h'] else False

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

    bs = load_data(args, SBM_path, 'bs')
    Bes = load_data(args, SBM_path, 'Bes')
    dls = load_data(args, SBM_path, 'dls')
    print(f'len of bs: {len(bs)}')

    if args.sbm in ['a', 'd', 'm', 'o']:
        pmode = posterior_modes(args, bs)
        modes = pmode.get_modes()
        L = total_evidence(pmode, bs, dls)

    if args.sbm in ['h']:
        pass

    save_data(args, SBM_path, bs, [], Bes, dls)
    save_modes(args, SBM_path, pmode, modes)
    save_evidence(args, SBM_path, L)
    print('saved all files')

    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()