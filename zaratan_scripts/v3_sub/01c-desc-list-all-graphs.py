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
    
    args.source = sys.argv[1]  #'allen'
    args.space =  sys.argv[2]  #'ccfv2'
    args.brain_div =  sys.argv[3]  #'whl'
    args.num_rois =   int(sys.argv[4])  #446
    args.resolution = int(sys.argv[5])  #200
    
    args.SEED = int(sys.argv[6]) # random seed
    
    return args

def main():
    args = setup_args()
    gt.seed_rng(args.SEED)
    np.random.seed(args.SEED)
    PARC_DESC = (
        f'source-{args.source}'
        f'_space-{args.space}'
        f'_braindiv-{args.brain_div}'
        f'_nrois-{args.num_rois}'
        f'_res-{args.resolution}'
    )
    
    BASE_path = f'{os.environ["HOME"]}/scratch/new_mouse_dataset'
    PARCELS_path = f'{BASE_path}/parcels'
    ROI_path = (
        f'{BASE_path}/roi-results-v3'
        f'/{PARC_DESC}'
    )
    os.system(f'mkdir -p {ROI_path}') 
    print(PARC_DESC)
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
# python 01d-desc-list-all-graphs.py spatial 225 True whl 162 100