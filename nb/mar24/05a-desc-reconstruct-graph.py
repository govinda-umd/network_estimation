import csv
import os
import sys
import numpy as np
import pandas as pd
import scipy as sp 
import pickle 
import copy

from scipy import sparse, stats
from scipy.special import gammaln
import glob
from tqdm import tqdm
import ants
from nipype.interfaces import afni
from itertools import combinations, permutations, product


import graph_tool.all as gt

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import colorcet as cc

plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams["errorbar.capsize"] = 0.5

import cmasher as cmr  # CITE ITS PAPER IN YOUR MANUSCRIPT

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)


# ==================

def setup_args():
    class ARGS():
        pass
    args = ARGS()
    
    args.TS_FILE = sys.argv[1]
    args.RECONST_method = sys.argv[2] #f'normal_dist'
    args.SEED = int(sys.argv[3]) 
    
    return args

def main():
    args = setup_args()
    gt.seed_rng(args.SEED)
    np.random.seed(args.SEED)
    
    fs = args.TS_FILE.split('/')
    GRAPH_path = '/'.join(fs[:-2] + ['graphs'])
    file = '_'.join(fs[-1].split('_')[:-1] + ['desc-graph.gt.gz'])
    print(fs[-1])
    
    ts = np.loadtxt(args.TS_FILE)
    if args.RECONST_method == f'normal_dist':
        state = gt.PseudoNormalBlockState(ts)
    if args.RECONST_method == f'lds':
        state = gt.LinearNormalBlockState(ts)
    
    # initial edge sweep
    ret = state.edge_mcmc_sweep(niter=10)
    delta = abs(ret[0])
    while delta > 1e-11:
        ret = state.mcmc_sweep(niter=10)
        delta = abs(ret[0])

    g = state.get_graph()
    A = gt.adjacency(state.get_graph()).todense()
    g.save(f'{GRAPH_path}/{file}')
    
    print(f'{GRAPH_path}/{file}')
    
    return None

# ==================

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
