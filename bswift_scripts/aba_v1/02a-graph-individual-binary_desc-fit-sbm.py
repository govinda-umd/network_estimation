import os
import sys
import numpy as np
import pandas as pd
import dill as pickle 
# import pickle
import pprint
from glob import glob
from scipy import sparse, stats
from scipy.special import gammaln
import re 
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

def load_graph(args, GRAPH_path):
    all_graphs = sorted(glob(f'{GRAPH_path}/*'))
    graph_file = all_graphs[args.GRAPH_IDX]
    g = gt.load_graph(graph_file)
    print(graph_file)
    return g, graph_file

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

def save_fit(args, SBM_path, state, bs, Bs, Bes, dls, pmode, modes, L):
    with open(f'{SBM_path}/desc-state.pkl', 'wb') as f:
        pickle.dump([state], f)
    
    with open(f'{SBM_path}/desc-partitions.pkl', 'wb') as f:
        pickle.dump([bs, Bs], f)
        
    with open(f'{SBM_path}/desc-Bes-dls.pkl', 'wb') as f:
        pickle.dump([Bes, dls], f)

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
    
    args.PARC_DESC = sys.argv[1] #"ABA_ROIs_final_gm_36"
    args.ANALYSIS = sys.argv[2] #"trial-end"
    args.GRAPH_DEF = sys.argv[3] #'constructed'
    args.GRAPH_METHOD = sys.argv[4] #'pearson'
    args.THRESHOLD = sys.argv[5] #'signed'
    args.EDGE_DEF = sys.argv[6] #'binary'
    args.EDGE_DENSITY = int(sys.argv[7]) #20
    args.LAYER_DEF = sys.argv[8] #'individual'
    args.DATA_UNIT = sys.argv[9] #'grp'
    args.GRAPH_IDX = int(sys.argv[10]) #0-3
    args.sbm = sys.argv[11] # a d h o m
    args.dc = sys.argv[12] == 'True'
    args.wait = int(sys.argv[13]) #1200
    args.total_samples = int(sys.argv[14]) #100,000
    args.B = int(sys.argv[15]) #1
    args.gamma = float(sys.argv[16]) #2.0
    args.SEED = int(sys.argv[17]) #100
    
    args.force_niter = args.total_samples
    
    return args

def main():
    args = setup_args()
    print(
        ', '.join([f'{key}: {value}' for key, value in vars(args).items()])
    )
    
    # gt.seed_rng(args.SEED)
    # np.random.seed(args.SEED)
    
    BASE_path = f'/data/homes/govindas/lab-data/aba'
    ROI_path = f'{BASE_path}/{args.PARC_DESC}'
    ROI_RESULTS_path = (
        f'{ROI_path}'
        f'/analysis-{args.ANALYSIS}'
        f'/graph-{args.GRAPH_DEF}/method-{args.GRAPH_METHOD}'
        f'/threshold-{args.THRESHOLD}/edge-{args.EDGE_DEF}/density-{args.EDGE_DENSITY}'
        f'/layer-{args.LAYER_DEF}/unit-{args.DATA_UNIT}'
    )
    GRAPH_path = f'{ROI_RESULTS_path}/graphs'

    g, graph_file = load_graph(args, GRAPH_path)
    match = re.search(r"cond-([^_]+)_([^_]+)_desc-", graph_file)
    if match:
        cond, name = match.groups()
        cond = '_'.join([cond, name]) 
    
    
    SBM_path = (
        f'{ROI_RESULTS_path}/model-fits'
        f'/{sbm_name(args)}/B-{args.B}/cond-{cond}'
    )
    os.system(f'mkdir -p {SBM_path}')
    
    state, state_args = create_state(args)
    args.num_draws = int((1/2) * args.total_samples) # remove burn-in period
    args.niter = 10
    print(state, state_args)
    
    if args.sbm in ['a', 'm', 'd', 'o']: # 'o' does not work for now.
        state = gt.minimize_blockmodel_dl(g, state=state, state_args=state_args)
        state, bs, Bes, Bs, dls, pmode, modes, L = fit_sbm(args, g, state)
        print(f'len of Bes: {len(Bes)}')

    if args.sbm in ['h']:
        state = gt.minimize_nested_blockmodel_dl(g, state=state, state_args=state_args)
        state, bs, Bes, Bs, dls, pmode, modes, L = fit_nested_sbm(args, g, state)
        print(f'len of Bes: {len(Bes)}')
    
    save_fit(args, SBM_path, state, bs, Bs, Bes, dls, pmode, modes, L)
    print('saved all files')

    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()