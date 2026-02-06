import csv
import os
import sys
import numpy as np
import pandas as pd
import scipy as sp 
import pickle 

from scipy import sparse, stats
from scipy.special import gammaln
import glob
from tqdm import tqdm

import graph_tool.all as gt

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)

def setup_args():
    class ARGS():
        pass
    args = ARGS()
    
    args.PARC_DESC = sys.argv[1] #'NEWMAX_ROIs_final_gm_100_2mm'
    args.ANALYSIS = sys.argv[2] #'trial-end'
    args.GRAPH_DEF = sys.argv[3] #'constructed'
    
    args.SEED = int(sys.argv[4]) # random seed
    
    return args

def main():
    args = setup_args()
    gt.seed_rng(args.SEED)
    np.random.seed(args.SEED)
    
    BASE_path = f'/data/homes/govindas/lab-data/aba'
    ROI_path = (
        f'{BASE_path}'
        f'/{args.PARC_DESC}'
        f'/analysis-{args.ANALYSIS}'
        f'/graph-{args.GRAPH_DEF}'
    )
    os.system(f'mkdir -p {ROI_path}') 
    print(ROI_path)
    
    GRAPH_FOLDERS = sorted(glob.glob(f'{ROI_path}/*/*/*/*/*/*/*', recursive=True))
    for FOLDER in GRAPH_FOLDERS:
        print(FOLDER)
        files = sorted(glob.glob(f'{FOLDER}/graphs/*', recursive=True))
        
        with open(f'{FOLDER}/all_graphs.txt', 'w') as f:
            f.writelines('\n'.join(files))
            
    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
        
# HOW TO RUN
# python 01c-desc-list-all-graphs.py NEWMAX_ROIs_final_gm_100_2mm trial-end constructed 100