#!/usr/bin/env python
# coding: utf-8

# # Sep 14, 2023: mouse whole brain fMRI: led stimulus regression brain maps 

# In[1]:


import csv
import os
import numpy as np
import pandas as pd
import scipy as sp 
import pickle 
from os.path import join as pjoin
from itertools import product
from tqdm import tqdm
from copy import deepcopy


# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import rainbow

plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams["errorbar.capsize"] = 0.5

import cmasher as cmr  # CITE ITS PAPER IN YOUR MANUSCRIPT

# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# In[2]:


class ARGS():
    pass

args = ARGS()

args.subs = np.arange(1, 11)
args.sess = np.arange(1, 4)
args.num_runs = 7

args.num_times = 600
args.num_rois = 6017 #  WILL CHANGE LATER


# In[3]:


stim_path = f'/home/govindas/mouse_dataset/stim'
censor_path = f'/home/govindas/mouse_dataset/roi/frame_censoring_mask'
data_path = f'/home/govindas/mouse_dataset/roi/data'
REG_path = f'/home/govindas/mouse_dataset/roi/regression_analysis'

for sub, ses in tqdm(product(args.subs, args.sess)):
    # stimulus----
    STIM = [['*'] for _ in range(args.num_runs)]
    stim_files = [
        f 
        for f in os.listdir(stim_path)
        if f'SLC{sub:02d}' in f
        if f'ses-{ses}' in f
    ]
    # stim_files, STIMS
    for stim_file in stim_files:
        idx = int([r for r in stim_file.split('_') if 'run' in r][0][-1]) - 1
        stim_times = pd.read_csv(f"{stim_path}/{stim_file}", index_col=0).dropna()['ledStim1Hz'].values
        l = list(np.where(np.diff(stim_times) == 1)[0]+1)
        STIM[idx] = l if len(l) > 0 else ['*']

    # censor and time series----
    DATA = [None for _ in range(args.num_runs)]
    CENSOR = [None for _ in range(args.num_runs)]
    for run in np.arange(1, args.num_runs+1, 1):
        DATA_ = np.zeros((args.num_times, args.num_rois))
        CENSOR_ = np.zeros((args.num_times,))
        censor_file = [
            f 
            for f in os.listdir(censor_path)
            if f'SLC{sub:02d}' in f
            if f'ses-{ses}' in f
            if f'run-{run}' in f
        ]
        data_file = [
            f 
            for f in os.listdir(data_path)
            if f'SLC{sub:02d}' in f
            if f'ses-{ses}' in f
            if f'run-{run}' in f
        ]
        censor_file = censor_file[0] if len(censor_file) > 0 else None
        if not os.path.isfile(f"{censor_path}/{censor_file}"):
            censor_times = None
            ts = None
            STIM[run-1] = ['*']
        else:
            censor_times = pd.read_csv(f"{censor_path}/{censor_file}").values.flatten()
            ts = np.load(f"{data_path}/{data_file[0]}").T # time x roi
            assert(len(np.where(censor_times)[0]) == len(ts))
            DATA_[np.where(censor_times)[0], :] = ts
            CENSOR_ = censor_times
        
        DATA[run-1] = DATA_ 
        CENSOR[run-1] = CENSOR_
    
    # print(f'sub {sub:02d}, ses {ses}')
    remove_runs = [i for i, x in enumerate(STIM) if x == ['*']]
    for run in sorted(remove_runs, reverse=True):
        del STIM[run]
        del DATA[run]
        del CENSOR[run]
    
    if len(STIM) == 0: continue
    
    # save CONCAT in a .1D file: per sub and per ses
    np.savetxt(
        f'{REG_path}/sub-SLC{sub:02d}_ses-{ses}_desc-CONCAT.1D', 
        np.arange(0, len(STIM)*args.num_times, args.num_times,),
        newline=' ', fmt='%d',
    )

    # save STIMS in a .txt file: per sub and per ses
    with open(f'{REG_path}/sub-SLC{sub:02d}_ses-{ses}_desc-STIM.txt', 'w', newline='') as f:
        wr = csv.writer(f, delimiter=' ')
        wr.writerows(STIM)
        
    # save DATA in a .1D file: per sub and per ses
    DATA = np.nan_to_num(np.vstack(DATA))
    np.savetxt(f'{REG_path}/sub-SLC{sub:02d}_ses-{ses}_desc-INPUT.1D', DATA)
    
    # save CENSOR in a .txt file: per sub and per ses
    CENSOR = np.nan_to_num(np.hstack(CENSOR))
    np.savetxt(f'{REG_path}/sub-SLC{sub:02d}_ses-{ses}_desc-CENSOR.txt', CENSOR)

