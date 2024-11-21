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
    args.type = sys.argv[1] #'spatial'
    args.roi_size = sys.argv[2] #225
    args.maintain_symmetry = sys.argv[3] #'True'
    args.brain_div = sys.argv[4] #'whl'
    args.num_rois = sys.argv[5] #162

    # graph
    args.GRAPH_DEF = sys.argv[6] #f'constructed'
    args.GRAPH_METHOD = sys.argv[7] #f'pearson-corr'
    args.THRESHOLDING = sys.argv[8] #f'positive'
    args.EDGE_DEF = sys.argv[9] #f'binary'
    args.EDGE_DENSITY = sys.argv[10] #10
    args.LAYER_DEF = sys.argv[11] #f'individual'
    args.DATA_UNIT = sys.argv[12] #f'ses'

    # sbm
    args.dc = sys.argv[13] == 'True' #True
    args.sbm = sys.argv[14] #'h'
    args.nested = args.sbm == 'h'
    args.force_niter = int(sys.argv[15]) #40000
    args.num_draws = int((4/5) * args.force_niter)
    args.num_samples = int(sys.argv[16]) #100 # from a chain for aggregating

    # animal
    args.sub = int(sys.argv[17]) # 1, 2, ... 10

    # random seed
    args.SEED = int(sys.argv[18]) 
    
    return args

def sbm_name(args):
    dc = f'dc' if args.dc else f'nd'
    dc = f'' if args.sbm in ['a'] else dc
    file = f'sbm-{dc}-{args.sbm}'
    return file

def collect_modes_single_chain(args, sbm_file):
    fs = sbm_file.split('/')

    g = gt.load_graph(f"{args.GRAPH_path}/{'_'.join([fs[-4], 'desc-graph.gt.gz'])}")

    with open(sbm_file, 'rb') as f:
        [modes] = pickle.load(f)

    dfs = []
    for idx, mode in enumerate(modes):
        df = pd.DataFrame({})
        for desc in fs[-4].split('_'): # sub ses
            D = desc.split('-')
            df[D[0]] = D[1:]

        D = fs[-3] # sbm
        df['sbm'] = D

        D = fs[-2].split('-') # B
        df[D[0]] = '_'.join(D[1:])

        df['graph'] = [g]
        b_hat = list(mode.get_max(g)) if not args.sbm in ['h'] else mode.get_max_nested()

        df['mode_id'] = [idx]
        df['mode'] = [mode]
        df['b_hat'] = [b_hat]
        df['omega'] = [np.round(mode.get_M()/args.num_draws, 3)]
        df['sigma'] = [np.round(mode.posterior_cdev(), 3)]
        ratio = df['omega'][0] / df['sigma'][0]
        ratio = ratio if not np.isnan(ratio) else 0.0
        df['ratio'] = [ratio]
        df['num_samples'] = [round(args.num_samples*df['omega'][0])]
        
        dfs += [df]
    dfs = pd.concat(dfs).reset_index(drop=True)
    return dfs

def collect_modes(args, sbm_files):
    sbm_dfs = []
    for sbm_file in tqdm(sbm_files):
        df = collect_modes_single_chain(args, sbm_file)
        sbm_dfs += [df]
    sbm_dfs = pd.concat(sbm_dfs).reset_index(drop=True)
    sbm_dfs = sbm_dfs.sort_values(['sub', 'ses', 'sbm', 'B'])
    return sbm_dfs

def sample_partitions(args, sbm_dfs):
    all_bs = []
    for idx, row in tqdm(sbm_dfs.iterrows()):
        bs = random.sample(
            list(row['mode'].get_partitions().values()), 
            row['num_samples']
        )
        all_bs += bs
        # all_bs += [
        #   row['mode'].sample_partition(MLE=True) 
        #   for _ in range(row['num_samples'])
        # ]
        # all_bs += [row['b_hat']]
    return all_bs

def sample_nested_partitions(args, sbm_dfs):
    all_bs = []
    for idx, row in tqdm(sbm_dfs.iterrows()):
        bs = random.sample(
            list(row['mode'].get_nested_partitions().values()), 
            row['num_samples']
        )
        bs = [gt.nested_partition_clear_null(b) for b in bs]
        all_bs += bs
        # all_bs += [
        #   row['mode'].sample_partition(MLE=True) 
        #   for _ in range(row['num_samples'])
        # ]
        # all_bs += [row['b_hat']]
    return all_bs

def posterior_modes(args, bs):
    pmode = gt.ModeClusterState(bs, nested=args.nested)
    gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    return pmode

def catalog_modes(args, cmode, g, all_bs):
    indiv_dfs = []
    for idx, mode in enumerate(cmode.get_modes()):
        b_hat = list(mode.get_max(g)) if not args.sbm in ['h'] else mode.get_max_nested()
        omega = np.round(mode.get_M()/len(all_bs), 3)
        sigma = np.round(mode.posterior_cdev(MLE=False), 3)
        ratio = omega / sigma
        ratio = np.round(ratio,3) if not np.isnan(ratio) else 0.0
        df = pd.DataFrame(dict(
            mode_id=[idx],
            mode=[mode],
            b_hat=[b_hat],
            omega=[omega],
            sigma=[sigma],
            ratio=[ratio],
        ))
        indiv_dfs += [df]
    indiv_dfs = pd.concat(indiv_dfs).reset_index(drop=True)
    return indiv_dfs

def individual_level_estimates(args, sub):
    print(f'sub {sub}')
    sbm_files = sorted(glob.glob(f'{args.SBM_path}/*{sub}*/{args.SBM}/*/desc-partition-modes.pkl', recursive=True))
    print(sbm_files)
    sbm_dfs = collect_modes(args, sbm_files)

    g = sbm_dfs.iloc[0]['graph'] # any graph with the same number of rois will do

    # sample b's from mode proportional to omega
    if args.sbm in ['h']:
        all_bs = sample_nested_partitions(args, sbm_dfs)
    elif args.sbm in ['a', 'd']:
        all_bs = sample_partitions(args, sbm_dfs)

    cmode = posterior_modes(args, all_bs)
    indiv_dfs = catalog_modes(args, cmode, g, all_bs)
    # TODO: align the modes once again with PartitionModeState()

    return indiv_dfs, sbm_dfs

def main():
    args = setup_args()
    # ---------
    args.PARC_DESC = (
        f'type-{args.type}'
        f'_size-{args.roi_size}'
        f'_symm-{args.maintain_symmetry}'
        f'_braindiv-{args.brain_div}'
        f'_nrois-{args.num_rois}'
    )
    args.BASE_path = f'{os.environ["HOME"]}/mouse_dataset/roi_results_v2'
    args.ROI_path = f'{args.BASE_path}/{args.PARC_DESC}'
    args.TS_path = f'{args.ROI_path}/runwise_timeseries'
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
    
    sub = f'SLC{args.sub:02d}'
    indiv_dfs, sbm_dfs = individual_level_estimates(args, sub=sub)

    folder = f'{args.ESTIM_path}/individual/sub-{sub}/partition-modes'
    os.system(f'mkdir -p {folder}')
    with open(f'{folder}/{args.SBM}_desc-df.pkl', 'wb') as f:
        pickle.dump(indiv_dfs, f)

    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()