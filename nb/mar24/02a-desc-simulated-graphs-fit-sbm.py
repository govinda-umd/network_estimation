import os
import sys
import numpy as np
import graph_tool.all as gt
import argparse
import ast
import pickle

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)

# ====================
def add_block(A, pos, size, density):
    # create block of given density
    nrows, ncols = size
    a = (np.random.rand(nrows,ncols)<density).astype('int')
    i, j = pos
    A[i : i+nrows, j:j+ncols] = a
    return A

def normalize_A(A):
    A = (A + A.T) / 2
    A -= np.diag(np.diag(A))
    return (A>0).astype('int')

def create_A(args,):
    args.num_rois = np.sum(args.block_sizes)
    A = np.zeros((args.num_rois, args.num_rois))

    pos = np.array([0,0])
    for block_size, block_density in zip(args.block_sizes, args.block_densities):
        A = add_block(A, pos, [block_size]*2, block_density)
        pos += block_size

    for ovp_size, ovp_density, ovp_pos in zip(
        args.overlap_sizes, args.overlap_densities, args.block_sizes[:-1]):
        pos = np.array([ovp_pos-ovp_size//2]*2)
        A += add_block(A, pos, [ovp_size]*2, ovp_density)
    A = normalize_A(A)
    return A

def make_graph_from_A(A):
    A = np.tril(A)
    edges = np.where(A)
    edge_list = list(zip(*(*edges, A[edges])))
    g = gt.Graph(
        edge_list, 
        eprops=[('weight', 'double')],
        directed=False
    )
    return g

# ====================

def mcmc_eq(args, g, state):
    bs = [] # partitions
    Bs = np.zeros(g.num_vertices() + 1) # number of partitions
    mdls = [] # entropies
    def collect_partitions(s):
        bs.append(s.b.a.copy())
        B = s.get_nonempty_B()
        Bs[B] += 1
        mdls.append(s.entropy())
    gt.mcmc_equilibrate(
        state, 
        wait=args.wait, 
        mcmc_args=dict(niter=args.niter), 
        callback=collect_partitions,
    )
    return state, bs, Bs, mdls

def get_partition_mode(args, g, state, bs, Bs):
    pmode = gt.PartitionModeState(bs, converge=True)
    return pmode

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
    
    if not state.overlap:
        marginals = v_marginals.copy()
    else:
        # average of probs. of half-edges incident on a vertex
        marginals = np.zeros((g.num_vertices(), v_marginals.shape[-1]))
        for v, hes in zip(g.iter_vertices(), state.half_edges):
            marginals[v] = np.mean(v_marginals[hes], axis=0)
    return marginals

def fit_sbm(args, g, state=gt.BlockState, state_args=dict()):
    state = gt.minimize_blockmodel_dl(g, state=state, state_args=state_args)
    print(f'heuristic complete')
    state, bs, Bs, mdls = mcmc_eq(args, g, state)
    print(f'mcmc complete')
    pmode = get_partition_mode(args, g, state, bs, Bs)
    marginal = get_marginals(args, g, state, pmode)
    print(f'marginals estimated')
    return marginal, pmode, state, bs, Bs, mdls

# ====================

def setup_args():
    parser = argparse.ArgumentParser()
    
    # graph construction
    parser.add_argument(
        '-bs',
        '--block_sizes',
        help='sizes of blocks',
        nargs="+",
        type=int,        
        default=[50, 50],
    )
    parser.add_argument(
        '-bd',
        '--block_densities',
        help='densities of blocks',
        nargs='+',
        type=float,
        default=[0.3, 0.3],
    )
    parser.add_argument(
        '-os',
        '--overlap_sizes',
        help='sizes of overlap',
        nargs='+',
        type=int,
        default=[6],
    )
    parser.add_argument(
        '-od',
        '--overlap_densities',
        help='densities of overlap',
        nargs='+',
        type=float,
        default=[0.3],
    )
    
    # sbm
    parser.add_argument(
        '-sbm',
        '--sbm_state',
        help='sbm state, gt.[Overlap/Nested]BlockState',
        type=str,
        default='d',
    )
    parser.add_argument(
        '-dc',
        '--deg_corr',
        help='degree corrected graph',
        type=ast.literal_eval,
        default=True,
    )
    parser.add_argument(
        '-w',
        '--wait',
        help='wait in mcmc equilibrate',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '-ni',
        '--niter',
        help='number of iterations to sample from posterior',
        type=int,
        default=10,
    )
    
    # general
    parser.add_argument(
        '-s',
        '--SEED',
        help='random seed',
        type=int,
        default=100,
    )
    
    return parser.parse_args()

def main():
    args = setup_args()
    
    SIM_path = f'{os.environ["HOME"]}/mouse_dataset/simulations/02a'
    os.system(f'mkdir -p {SIM_path}')
    
    gt.seed_rng(args.SEED)
    np.random.seed(args.SEED)
    
    print(args)

    # create graph
    A = create_A(args)   
    g = make_graph_from_A(A)

    graph_density = A.sum() / np.prod(A.shape)
    print(f'density: {graph_density}')
    print(g)
    
    # fit sbm
    state_args = dict(deg_corr=args.deg_corr)
    if args.sbm_state == 'd': # disjoint
        state = gt.BlockState
    elif args.sbm_state == 'o':
        state = gt.OverlapBlockState
    (
        marginal, pmode, 
        state, bs, Bs, mdls
    )= fit_sbm(args, g, state, state_args)
    
    file_name = (
        f'bs-{args.block_sizes}'
        f'_bd-{args.block_densities}'
        f'_os-{args.overlap_sizes}'
        f'_od-{args.overlap_densities}'
        f'_sbm-{args.sbm_state}'
        f'_dc-{args.deg_corr}'
        f'.npy'
    )
    file = f'{SIM_path}/{file_name}'
    print(f'{file}')
    with open(f'{file}', 'wb') as f:
        pickle.dump([marginal, pmode, g, state, bs, Bs, mdls], f)
    print(f'saved to {file}')
    
# ====================

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()