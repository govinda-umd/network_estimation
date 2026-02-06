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

def save_state(args, SBM_path, state):
    with open(f'{SBM_path}/desc-state.pkl', 'wb') as f:
        pickle.dump([state], f)
    return None

def clear_data(args, SBM_path):
    # clear the folder for a fresh start
    os.system(f'rm -rf {SBM_path}/*')
    return None

def check_diversity(Bes, length):
    """
    Returns True if the Be sequence is diverse (not stuck),
    based on both convergence flatness and histogram entropy.
    Requires both: NOT converged AND entropy above threshold.
    """
    if len(Bes) < length:
        return False

    recent = np.array(Bes[-length:])

    # --- Convergence check ---
    diffs = np.abs(np.diff(recent))
    max_delta = np.max(diffs)
    var = np.var(recent)
    cv = np.std(recent) / (np.mean(recent) + 1e-6)
    not_converged = not (
        max_delta < 2 and
        var < 1e-2 and
        cv < 0.01
    )

    # --- Entropy check ---
    values, counts = np.unique(recent, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(len(probs))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    entropy_diverse = norm_entropy > 0.5

    # --- Final decision: BOTH must be true ---
    is_diverse = not_converged and entropy_diverse

    print(f"   diversity check:")
    print(f"     maxΔ = {max_delta:.3f}, var = {var:.4f}, cv = {cv:.4f} → not_converged: {not_converged}")
    print(f"     norm_entropy = {norm_entropy:.3f} → entropy_diverse: {entropy_diverse}")
    print(f"     → overall: {'diverse' if is_diverse else 'stuck'}")

    return is_diverse

def get_initial_state(args, g, diverse=False):

    attempt = 0
    while not diverse:
        state, state_args = create_state(args)
        state = gt.minimize_blockmodel_dl(g, state=state, state_args=state_args)
        state, bs, Bs, Bes, dls = mcmc_eq(args, g, state)
        diverse = check_diversity(Bes, args.num_draws)
        print(f'attempt {attempt} unsuccessful')
        attempt += 1
        # diverse = True # COMMENT THIS LINE DURING MAIN EXECUTION

    print(f'attempt {attempt} successful')
    return state, Bes

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

    # mcmc_eq() parameters used during diversity checking
    parser.add_argument('--wait', type=int, default=1200,
                        help='Wait time for mcmc_equilibrate()')
    parser.add_argument('--force-niter', type=int, default=50,
                        help='Force niter for mcmc_equilibrate()')
    parser.add_argument('--niter', type=int, default=10,
                        help='Number of iterations between samples to reduce auto-correlation')

    args = parser.parse_args()

    # Derived quantity
    args.num_draws = int(np.round(0.4 * args.force_niter))
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

    print(f'starting heuristic')

    clear_data(args, SBM_path)
    
    if args.sbm in ['a', 'm', 'd', 'o']:
        args.B = np.random.randint(25, 100) # quick fix, change later
        state, Bes = get_initial_state(args, g, diverse=False)
        print(f'len of Bes: {len(Bes)}')
    
    if args.sbm in ['h']:
        pass
        # IMPLEMENT WITH MINIMIZE_NESTED_BLOCKMODEL_DL()

    save_state(args, SBM_path, state)
    print('saved all files')

    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()