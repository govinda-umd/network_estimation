import csv
import os
import sys
import numpy as np
import pandas as pd
import dill as pickle 
from tqdm import tqdm
import glob
import random
import re

import graph_tool.all as gt

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)

def set_seed(args):
    gt.seed_rng(args.SEED)
    np.random.seed(args.SEED)

def sbm_name(args):
    dc = f'dc' if args.dc else f'nd'
    dc = f'' if args.sbm in ['m', 'a'] else dc
    file = f'sbm-{dc}-{args.sbm}'
    return file

def get_pi_matrix(args, mrgnls):
    num_comms = np.max([len(mrgnl) for mrgnl in mrgnls])
    pi = np.zeros((len(mrgnls), num_comms))

    for idx_node, mrgnl in enumerate(mrgnls):
        mrgnl = np.array(mrgnl)
        pi[idx_node, np.where(mrgnl)[0]] = mrgnl[mrgnl > 0]

    pi = pi / np.expand_dims(pi.sum(axis=-1), axis=-1)
    return pi # marginals matrix

def collect_sbm_files(args, sbm_files, g):
    sbms_df = []
    for file in tqdm(sbm_files):
        with open(file, 'rb') as f:
            [modes] = pickle.load(f)
        
        match = re.search(r"B-(\d+)", file)
        if match:
            B, = match.groups()
            B = int(B)

        M = np.sum([mode.get_M() for mode in modes]) # total samples

        for mode_id, mode in enumerate(modes):
            omega = np.round(mode.get_M() / M, 3)
            sigma = np.round(mode.posterior_cdev(), 3)
            ratio = omega / sigma
            ratio = np.round(ratio, 3) if not np.isnan(ratio) else 0.0
            
            mrgnls = mode.get_marginal(g) #TODO: modify to get marginals for higher levels in hSBM
            pi = get_pi_matrix(args, mrgnls)
        
            row = pd.DataFrame(dict(
                cond=[args.COND],
                sbm=[args.SBM],
                B=[B],
                mode_id=[mode_id],
                mode=[mode],
                omega=[omega],
                sigma=[sigma],
                ratio=[ratio],
                pi=[pi],
            ))
            sbms_df += [row]
        # break
    sbms_df = pd.concat(sbms_df).reset_index(drop=True)
    return sbms_df

def sample_partitions(args, sbms_df):
    all_bs_df = []
    for idx, row in tqdm(sbms_df.iterrows()):
        bs = random.sample(
            list(row['mode'].get_partitions().values()), 
            row['num_samples']
        )
        mode_ids = [idx]*len(bs)
        df = pd.DataFrame(dict(
            mode_id=mode_ids,
            b=bs,
        ))
        all_bs_df += [df]
        # all_bs += [
        #   row['mode'].sample_partition(MLE=True) 
        #   for _ in range(row['num_samples'])
        # ]
    all_bs_df = pd.concat(all_bs_df).reset_index(drop=True)
    return all_bs_df

def sample_nested_partitions(args, sbm_dfs):
    all_bs_df = []
    for idx, row in tqdm(sbm_dfs.iterrows()):
        bs = random.sample(
            list(row['mode'].get_nested_partitions().values()), 
            row['num_samples']
        )
        # bs = [gt.nested_partition_clear_null(b) for b in bs]
        mode_ids = [idx]*len(bs)
        df = pd.DataFrame(dict(
            mode_id=mode_ids,
            b=bs,
        ))
        all_bs_df += [df]
        # all_bs += [
        #   row['mode'].sample_partition(MLE=True) 
        #   for _ in range(row['num_samples'])
        # ]
    all_bs_df = pd.concat(all_bs_df).reset_index(drop=True)
    return all_bs_df

def posterior_modes(args, bs):
    cmode = gt.ModeClusterState(bs, nested=args.nested, relabel=False)
    gt.mcmc_equilibrate(cmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    cmode.relabel(maxiter=1000)
    return cmode

def catalog_indiv_modes(args, cmode, g):
    cmode.relabel(maxiter=1000)
    modes = cmode.get_modes()
    pis = [get_pi_matrix(args, mode.get_marginal(g))  for mode in modes]
    M = np.sum([mode.get_M() for mode in modes])
    omegas = [mode.get_M() / M for mode in modes]
    sigmas = [mode.posterior_cdev() for mode in modes]
    conds = [args.COND]*len(omegas)
    sbms = [args.SBM]*len(omegas)
    modes_df = pd.DataFrame(dict(
        cond=conds, sbm=sbms, 
        mode_id=np.arange(len(pis)), mode=modes, 
        pi=pis, omega=omegas, sigma=sigmas
    ))
    return modes_df

def indiv_level_modes(args, g):
    sbm_files = sorted(glob.glob(f'{args.SBM_path}/desc-partition-modes.pkl', recursive=True))
    sbms_df = collect_sbm_files(args, sbm_files, g)

    # sample partitions per mode
    sbms_df['num_samples'] = sbms_df['omega'].apply(lambda x: np.round(x * args.total_samples).astype(int))
    if args.sbm in ['m', 'a', 'd', 'o']:
        all_bs_df = sample_partitions(args, sbms_df)
    if args.sbm in ['h']:
        all_bs_df = sample_nested_partitions(args, sbms_df)

    # align all samples iteratively until the labels converge
    pmode = gt.PartitionModeState(all_bs_df['b'], relabel=True, nested=args.nested, converge=False)
    ent_diff = -np.inf
    while ent_diff < 1e-10:
        ed = pmode.replace_partitions()
        print(ed)
        if np.isclose(ed, ent_diff, rtol=1e-10):
            break
        ent_diff = ed

    if args.sbm in ['m', 'a', 'd', 'o']:
        bs = pmode.get_partitions()
    if args.sbm in ['h']:
        bs = pmode.get_nested_partitions()
    bs = {k:v for k, v in sorted(bs.items())}
    all_bs_df['b_aligned'] = list(bs.values())

    # find modes per animal using these aligned samples
    cmode = posterior_modes(args, all_bs_df['b_aligned'].to_list())

    # catalog/collect all the node marginals per mode
    modes_df = catalog_indiv_modes(args, cmode, g)

    return modes_df

def load_graph(args, GRAPH_path):
    all_graphs = sorted(glob.glob(f'{GRAPH_path}/*'))
    graph_file = all_graphs[args.GRAPH_IDX]
    g = gt.load_graph(graph_file)
    print(graph_file)
    return g, graph_file

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
    args.COND = sys.argv[10] #'highT'
    args.GRAPH_IDX = int(sys.argv[11]) #0-3
    args.sbm = sys.argv[12] # a d h o m
    args.dc = sys.argv[13] == 'True'
    args.force_niter = int(sys.argv[14]) #100,000
    args.total_samples = int(sys.argv[15]) #10,000 from a chain for aggregating
    args.gamma = float(sys.argv[16]) #2.0
    args.SEED = int(sys.argv[17]) #100

    args.force_niter = args.total_samples
    args.nested = args.sbm == 'h'
    args.num_draws = int((1/2) * args.force_niter)
    
    return args

def main():
    args = setup_args()
    print(
        ', '.join([f'{key}: {value}' for key, value in vars(args).items()])
    )
    
    gt.seed_rng(args.SEED)
    np.random.seed(args.SEED)
    
    # ---------
    args.BASE_path = f'/data/homes/govindas/lab-data/aba'
    args.ROI_path = f'{args.BASE_path}/{args.PARC_DESC}'
    args.TS_path = f'{args.ROI_path}/roi_timeseries'
    args.ROI_RESULTS_path = (
        f'{args.ROI_path}'
        f'/analysis-{args.ANALYSIS}'
        f'/graph-{args.GRAPH_DEF}/method-{args.GRAPH_METHOD}'
        f'/threshold-{args.THRESHOLD}/edge-{args.EDGE_DEF}/density-{args.EDGE_DENSITY}'
        f'/layer-{args.LAYER_DEF}/unit-{args.DATA_UNIT}/cond-{args.COND}'
    )
    args.GRAPH_path = f'{args.ROI_RESULTS_path}/graphs'
    print(args.GRAPH_path)
    args.SBM = sbm_name(args)
    
    g, graph_file = load_graph(args, args.GRAPH_path)
    match = re.search(r'sub-([^_]+)', graph_file)
    if match:
        sub = match.group(1)
    
    args.SBM_path = (
        f'{args.ROI_RESULTS_path}/model-fits'
        f'/sub-{sub}'
        f'/{args.SBM}/B-*'
    ) 
    args.ESTIM_path = (
        f'{args.ROI_RESULTS_path}/estimates/individual'
        f'/sub-{sub}'
    )
    os.makedirs(f'{args.ESTIM_path}', exist_ok=True)
    print(f'{args.ESTIM_path}')
    

    

    # ---------
    modes_df = indiv_level_modes(args, g)

    folder = f'{args.ESTIM_path}/partition-modes'
    os.system(f'mkdir -p {folder}')
    with open(f'{folder}/{args.SBM}_desc-df.pkl', 'wb') as f:
        pickle.dump(modes_df, f)
    
    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()