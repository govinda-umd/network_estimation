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

from multiprocessing import Pool
from functools import partial

from tqdm import tqdm

import argparse

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)

def clear_data(args, SBM_path):
    # clear the folder for a fresh start
    os.system(f'rm -rf {SBM_path}/*')
    return None

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
        dict(entropy_args=dict(gamma=args.gamma)),
    ]
    state, state_args = state_df[args.sbm]
    return state, state_args

def sbm_name(args):
    dc = f'dc' if args.dc else f'nd'
    dc = f'' if args.sbm in ['a', 'm'] else dc
    file = f'sbm-{dc}-{args.sbm}'
    return file

# def collect_partitions(args, g):
#     '''
#     using heuristic repeatedly
#     '''
#     bs = [] # partitions
#     Bs = np.zeros(g.num_vertices() + 1) # number of blocks
#     Bes = [] # number of effective blocks
#     dls = [] # description length history
#     for idx in tqdm(range(args.force_niter)):
#         state, state_args = create_state(args)
#         state = gt.minimize_blockmodel_dl(g, state=state, state_args=state_args)
#         bs.append(state.b.a.copy())
#         B = state.get_nonempty_B()
#         Bs[B] += 1
#         Bes.append(state.get_Be())
#         dls.append(state.entropy())

#     return state, bs, Bs, Bes, dls

def run_single_heuristic(args, g, _):
    state, state_args = create_state(args)
    state = gt.minimize_blockmodel_dl(g, state=state, state_args=state_args)
    b = state.b.a.copy()
    B = state.get_nonempty_B()
    Be = state.get_Be()
    dl = state.entropy()
    return state, b, B, Be, dl

def collect_partitions(args, g):
    """
    Use multiprocessing to run `minimize_blockmodel_dl()` in parallel.
    Each run is independent and returns (b, B, Be, dl).
    """
    bs = []
    Bs = np.zeros(g.num_vertices() + 1)
    Bes = []
    dls = []

    func = partial(run_single_heuristic, args, g)
    with Pool(processes=10) as pool:
        results = list(tqdm(pool.imap(func, range(args.force_niter)), total=args.force_niter))

    for state, b, B, Be, dl in results:
        bs.append(b)
        Bs[B] += 1
        Bes.append(Be)
        dls.append(dl)

    return state, bs, Bs, Bes, dls

def save_run(args, SBM_path, state, bs, Bs, Bes, dls):
    with open(f'{SBM_path}/segment-{args.segment_id}_desc-state.pkl', 'wb') as f:
        pickle.dump([state], f)

    with open(f'{SBM_path}/segment-{args.segment_id}_desc-partitions.pkl', 'wb') as f:
        pickle.dump([bs, Bs, Bes, dls], f)
    
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
    parser.add_argument('--seed', type=int, default=100,
                        help='Random seed for reproducibility')
    
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
    args.force_niter = args.segment_len #+ 1
    args.num_draws = int(np.round(0.5 * args.force_niter))
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

    # clear_data(args, SBM_path)
    # print(f'cleared previous clutter')

    print(f'starting heuristic')
    
    if args.sbm in ['a', 'm', 'd', 'o']:
        args.B = np.random.randint(50, 200) # quick fix, change later
        state, bs, Bs, Bes, dls = collect_partitions(args, g)
        print(f'len of Bes: {len(Bes)}')
    
    if args.sbm in ['h']:
        pass
        # IMPLEMENT WITH MINIMIZE_NESTED_BLOCKMODEL_DL()

    save_run(args, SBM_path, state, bs, Bs, Bes, dls)
    print('saved all files')

    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()