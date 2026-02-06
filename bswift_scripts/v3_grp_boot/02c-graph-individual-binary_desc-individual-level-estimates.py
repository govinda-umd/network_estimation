import csv
import os
import sys
import numpy as np
import pandas as pd
import dill as pickle 
from tqdm import tqdm
import glob
import random

import graph_tool.all as gt

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)

def setup_args():
    class ARGS():
        pass
    args = ARGS()

    # parcellation
    args.PARC_DESC = sys.argv[1]

    # graph
    args.GRAPH_DEF = sys.argv[2] #f'constructed'
    args.GRAPH_METHOD = sys.argv[3] #f'pearson'
    args.THRESHOLDING = sys.argv[4] #f'signed'
    args.EDGE_DEF = sys.argv[5] #f'binary'
    args.EDGE_DENSITY = sys.argv[6] #20
    args.LAYER_DEF = sys.argv[7] #f'individual'
    args.DATA_UNIT = sys.argv[8] #f'sub'

    # sbm
    args.dc = sys.argv[9] == 'True' #True
    args.sbm = sys.argv[10] #'h'
    args.nested = args.sbm == 'h'
    args.force_niter = int(sys.argv[11]) #40000
    args.num_draws = int((1/2) * args.force_niter)
    args.total_samples = int(sys.argv[12]) #1000 # from a chain for aggregating

    # animal
    args.sub = int(sys.argv[13]) # 0, ..., 99

    # random seed
    args.SEED = int(sys.argv[14]) 
    
    return args

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
        
        fs = file.split('/')
        sub = '-'.join([s for s in fs if 'boot-' in s][0].split('-')[1:])
        sbm = [s for s in fs if 'sbm-' in s][0]
        B = [s for s in fs if 'B-' in s][0].split('-')[-1]

        M = np.sum([mode.get_M() for mode in modes]) # total samples

        for mode_id, mode in enumerate(modes):
            omega = np.round(mode.get_M() / M, 3)
            sigma = np.round(mode.posterior_cdev(), 3)
            ratio = omega / sigma
            ratio = np.round(ratio, 3) if not np.isnan(ratio) else 0.0
            
            mrgnls = mode.get_marginal(g) #TODO: modify to get marginals for higher levels in hSBM
            pi = get_pi_matrix(args, mrgnls)
        
            row = pd.DataFrame(dict(
                sub=[sub],
                sbm=[sbm],
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
    gt.mcmc_equilibrate(cmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf), verbose=True)
    cmode.relabel(maxiter=1000)
    return cmode

def catalog_indiv_modes(args, cmode, g, sub):
    cmode.relabel(maxiter=1000)
    modes = cmode.get_modes()
    pis = [get_pi_matrix(args, mode.get_marginal(g))  for mode in modes]
    M = np.sum([mode.get_M() for mode in modes])
    omegas = [mode.get_M() / M for mode in modes]
    sigmas = [mode.posterior_cdev() for mode in modes]
    subs = [sub]*len(omegas)
    sbms = [args.SBM]*len(omegas)
    modes_df = pd.DataFrame(dict(boot=subs, sbm=sbms, mode_id=np.arange(len(pis)), mode=modes, pi=pis, omega=omegas, sigma=sigmas))
    return modes_df

def indiv_level_modes(args, sub):
    gfile = sorted(glob.glob(f'{args.GRAPH_path}/boot-{sub}*', recursive=True))[0]
    g = gt.load_graph(gfile)

    sbm_files = sorted(glob.glob(f'{args.SBM_path}/boot-{sub}/{args.SBM}/*/desc-partition-modes.pkl', recursive=True))
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
    modes_df = catalog_indiv_modes(args, cmode, g, sub)

    return modes_df

def main():
    args = setup_args()
    # ---------
    args.BASE_path = f'/data/homes/govindas/new_mouse_dataset/roi-results-v3'
    args.ROI_path = f'{args.BASE_path}/{args.PARC_DESC}'
    args.TS_path = f'{args.ROI_path}/roi_timeseries'
    args.ROI_RESULTS_path = (
        f'{args.ROI_path}'
        f'/graph-{args.GRAPH_DEF}/method-{args.GRAPH_METHOD}'
        f'/threshold-{args.THRESHOLDING}/edge-{args.EDGE_DEF}/density-{args.EDGE_DENSITY}'
        f'/layer-{args.LAYER_DEF}/unit-{args.DATA_UNIT}'
    )
    args.GRAPH_path = f'{args.ROI_RESULTS_path}/graphs'
    os.system(f'mkdir -p {args.GRAPH_path}')
    args.SBM_path = f'{args.ROI_RESULTS_path}/model-fits'
    os.system(f'mkdir -p {args.SBM_path}')
    args.ESTIM_path = f'{args.ROI_RESULTS_path}/estimates'
    os.system(f'mkdir -p {args.ESTIM_path}/individual')
    os.system(f'mkdir -p {args.ESTIM_path}/group')

    args.SBM = sbm_name(args)

    # ---------
    set_seed(args)
    sub = f'{args.sub:03d}'
    print(f'{args.ESTIM_path}/individual/boot-{sub}')

    # ---------
    modes_df = indiv_level_modes(args, sub)

    folder = f'{args.ESTIM_path}/individual/boot-{sub}/partition-modes'
    os.system(f'mkdir -p {folder}')
    with open(f'{folder}/{args.SBM}_desc-df.pkl', 'wb') as f:
        pickle.dump(modes_df, f)
    
    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()