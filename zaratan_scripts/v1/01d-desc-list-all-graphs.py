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
    
    args.type = sys.argv[1] #'spatial'
    args.roi_size = int(sys.argv[2]) #225
    args.maintain_symmetry = sys.argv[3] == 'True' #True
    args.brain_div = sys.argv[4] #'whl'
    args.num_rois = int(sys.argv[5]) #162
    
    args.SEED = int(sys.argv[6]) # random seed
    
    return args


def main():
    args = setup_args()
    gt.seed_rng(args.SEED)
    np.random.seed(args.SEED)
    DESC = (
        f'type-{args.type}'
        f'_size-{args.roi_size}'
        f'_symm-{args.maintain_symmetry}'
        f'_braindiv-{args.brain_div}'
        f'_nrois-{args.num_rois}'
    )

    BASE_path = f'{os.environ["HOME"]}/scratch/mouse_dataset/roi_results'
    print(BASE_path)
    print(DESC)

    
    denst_folders = sorted(glob.glob(f'{BASE_path}/{DESC}/*wise/*', recursive=True))
    for denst_folder in denst_folders:
        print(denst_folder)
        files = sorted(glob.glob(f'{denst_folder}/corr_mats/*', recursive=True))
        
        with open(f'{denst_folder}/all_graphs.txt', 'w',) as f:
            f.writelines('\n'.join(files))

    return None

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

# HOW TO RUN
# python 01d-desc-list-all-graphs.py spatial 225 1 whl 162 100