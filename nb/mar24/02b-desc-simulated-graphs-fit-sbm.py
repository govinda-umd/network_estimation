import os
import sys
import numpy as np
import pandas as pd 
import networkx as nx
import graph_tool.all as gt
import pickle

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)

# ======================

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
    
    if (not args.sbm == 'p') and (state.overlap):
        # average of probs. of half-edges incident on a vertex
        marginals = np.zeros((g.num_vertices(), v_marginals.shape[-1]))
        for v, hes in zip(g.iter_vertices(), state.half_edges):
            marginals[v] = np.mean(v_marginals[hes], axis=0)
    else:
        marginals = v_marginals.copy()
    return marginals

def fit_sbm(args, g, state=gt.BlockState, state_args=dict()):
    state = gt.minimize_blockmodel_dl(
        g, state=state, 
        state_args=state_args, 
        # multilevel_mcmc_args=dict(B_min=args.B, B_max=args.B)
    )
    print(f'heuristic complete')
    state, bs, Bs, mdls = mcmc_eq(args, g, state)
    print(f'mcmc complete')
    pmode = get_partition_mode(args, g, state, bs, Bs)
    marginal = get_marginals(args, g, state, pmode)
    print(f'marginals estimated')
    return marginal, pmode, state, bs, Bs, mdls

# ======================

def add_block(A, pos, size, density, diag=True):
    # create block of given density
    nrows, ncols = size
    i, j = pos
    if not diag:
        a = (np.random.rand(nrows,ncols)<density).astype('int')
        A[i:i+nrows, j:j+ncols] = a
    if diag:
        A_ = np.zeros((nrows, ncols))
        a = (np.random.rand(nrows//2, ncols//2)<density).astype('int')
        A_[nrows//2:nrows, 0:ncols//2] = a
        A_ = normalize_A(A_)
        A[i:i+nrows, j:j+ncols] = A_
    
    return A

def normalize_A(A):
    A = (A + A.T) / 2
    A -= np.diag(np.diag(A))
    return (A>0).astype('int')

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


def create_graph(args):
    if args.gn == 1: # graph_number
        # graph1
        sizes = [22, 6, 22, 22, 6, 22]
        p = [
            [0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.7, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.7, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.3, 0.1],
        ]
        G = nx.stochastic_block_model(
            sizes=sizes, 
            p=p,
            seed=args.SEED,
        )
        A = nx.adjacency_matrix(G).todense()
        g = make_graph_from_A(A)
        
    if args.gn == 2:
        # graph2
        sizes = [22, 6, 22, 22, 6, 22]
        p = [
            [0.3, 0.3, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.1, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.3, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.3],
        ]
        G = nx.stochastic_block_model(
            sizes=sizes, 
            p=p,
            seed=args.SEED,
        )
        A = nx.adjacency_matrix(G).todense()
        g = make_graph_from_A(A)

    if args.gn == 3:
        # graph3
        sizes = [22, 6, 22, 6, 22, 22]
        p = [
            [0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.3, 0.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.3, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.3, 0.2, 0.2],
            [0.0, 0.0, 0.0, 0.2, 0.3, 0.0],
            [0.0, 0.2, 0.0, 0.2, 0.0, 0.3],
        ]
        G = nx.stochastic_block_model(
            sizes=sizes, 
            p=p,
            seed=args.SEED,
        )
        A = nx.adjacency_matrix(G).todense()
        g = make_graph_from_A(A)
        
    if args.gn == 4:
        # graph4
        sizes = [22, 6, 22, 6, 22, 6, 22]
        p = [
            [0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.3, 0.2, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2, 0.3, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.2, 0.3, 0.2, 0.0],
            [0.0, 0.0, 0.2, 0.0, 0.2, 0.3, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3],
        ]
        G = nx.stochastic_block_model(
            sizes=sizes, 
            p=p,
            seed=args.SEED,
        )
        A = nx.adjacency_matrix(G).todense()
        g = make_graph_from_A(A)
    
    if args.gn == 5:
        # graph5
        sizes = [25, 25, 25, 25]
        p = [
            [0.3, 0.01, 0.01, 0.01],
            [0.01, 0.3, 0.01, 0.01],
            [0.01, 0.01, 0.3, 0.01],
            [0.01, 0.01, 0.01, 0.3],
        ]
        G = nx.stochastic_block_model(
            sizes=sizes, 
            p=p,
            seed=args.SEED,
        )
        A = nx.adjacency_matrix(G).todense()
        A = add_block(A, (20,20), (10,10), 0.3)
        A = add_block(A, (44,44), (10,10), 0.3)
        A = add_block(A, (70,70), (10,10), 0.3)
        A = add_block(A, (0,94), (5,5), 0.3, diag=False)
        A = add_block(A, (94,0), (5,5), 0.3, diag=False)
        g = make_graph_from_A(A)
    
    if args.gn == 6:
        # graph6
        sizes = [25, 25, 25, 25]
        p = [
            [0.3, 0.01, 0.01, 0.01],
            [0.01, 0.3, 0.01, 0.01],
            [0.01, 0.01, 0.3, 0.01],
            [0.01, 0.01, 0.01, 0.3],
        ]
        G = nx.stochastic_block_model(
            sizes=sizes, 
            p=p,
            seed=args.SEED,
        )
        A = nx.adjacency_matrix(G).todense()
        A = add_block(A, (20,20), (10,10), 0.3)
        A = add_block(A, (44,44), (10,10), 0.3)
        A = add_block(A, (70,70), (10,10), 0.3)
        A = normalize_A(A)
        g = make_graph_from_A(A)
        
    if args.gn == 7:
        # graph7
        sizes = [25, 25, 25, 25]
        p = [
            [0.1, 0.01, 0.01, 0.01],
            [0.01, 0.3, 0.01, 0.01],
            [0.01, 0.01, 0.3, 0.01],
            [0.01, 0.01, 0.01, 0.1],
        ]
        G = nx.stochastic_block_model(
            sizes=sizes, 
            p=p,
            seed=args.SEED,
        )
        A = nx.adjacency_matrix(G).todense()
        A = add_block(A, (20,20), (10,10), 0.3)
        A = add_block(A, (44,44), (10,10), 0.1)
        A = add_block(A, (70,70), (10,10), 0.3)
        A = normalize_A(A)
        g = make_graph_from_A(A)
        
    return g

# ======================

def setup_args():
    class ARGS():
        pass
    args = ARGS()
    
    args.gn = int(sys.argv[1])
    args.sbm = sys.argv[2]
    args.dc = sys.argv[3] == '1'
    args.SEED = int(sys.argv[4])
    
    return args

def main():
    args = setup_args()
    print(
        args.gn, args.sbm,
        args.dc, args.SEED
    )
    
    SIM_path = f'{os.environ["HOME"]}/mouse_dataset/simulations/02b'
    os.system(f'mkdir -p {SIM_path}')
    
    gt.seed_rng(args.SEED)
    np.random.seed(args.SEED)
    
    args.wait = 1000
    args.niter = 10
    
    g = create_graph(args,)
    state_df = pd.DataFrame({
        'sbm': ['d', 'o', 'p'],
        'state': [gt.BlockState, gt.OverlapBlockState, gt.PPBlockState],
    })
    state = state_df[state_df['sbm'] == args.sbm]['state'].to_list()[0]
    state_args = dict(deg_corr=args.dc) if not args.sbm == 'p' else dict()
    print(state_args)
    marginal, pmode, state, bs, Bs, mdls = fit_sbm(
        args, g,
        state, state_args
    )
    
    file_name = (
        f'gn-{args.gn}'
        f'_sbm-{args.sbm}'
        f'_dc-{args.dc}'
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