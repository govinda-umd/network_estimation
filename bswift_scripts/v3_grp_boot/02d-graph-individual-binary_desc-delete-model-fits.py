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
    args.DATA_UNIT = sys.argv[8] #f'grp'

    # sbm
    args.dc = sys.argv[9] == 'True' #True
    args.sbm = sys.argv[10] #'h'
    args.nested = args.sbm == 'h'

    # animal
    args.sub = int(sys.argv[11]) # 0, ..., 99

    # random seed
    args.SEED = int(sys.argv[12]) 
    
    return args

def set_seed(args):
    gt.seed_rng(args.SEED)
    np.random.seed(args.SEED)

def sbm_name(args):
    dc = f'dc' if args.dc else f'nd'
    dc = f'' if args.sbm in ['m', 'a'] else dc
    file = f'sbm-{dc}-{args.sbm}'
    return file


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
    sub = f'{args.sub:02d}'
    print(f'{args.SBM_path}/boot-{sub}')

    # ---------
    sbm_files = sorted(glob.glob(f'{args.SBM_path}/boot-{sub}/{args.SBM}/*/p*.pkl', recursive=True))
    print(sbm_files)

    for file in sbm_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass
    
    
    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()