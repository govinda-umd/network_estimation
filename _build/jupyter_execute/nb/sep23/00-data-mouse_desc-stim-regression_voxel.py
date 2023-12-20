#!/usr/bin/env python
# coding: utf-8

# # Sep 18, 2023: mouse whole brain fMRI, voxel level data: led stimulus regression brain maps 

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
from pathlib import Path
import subprocess

# nilearn
from nilearn import image

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
args.space_size = [58, 79, 45]


# In[3]:


stim_path = f'/home/govindas/mouse_dataset/stim'
censor_path = f'/home/govindas/mouse_dataset/voxel/frame_censoring_mask'
mask_path = f'/home/govindas/mouse_dataset/voxel/commonspace_mask'
data_path = f'/home/govindas/mouse_dataset/voxel/cleaned_timeseries'
REG_path = f'/home/govindas/mouse_dataset/voxel/regression_analysis'

def get_stim(sub, ses):
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
    return STIM

def get_censor_times(sub, ses, run):
    try:
        censor_files = [
            f 
            for f in os.listdir(censor_path)
            if f'SLC{sub:02d}' in f
            if f'ses-{ses}' in f
            if f'run-{run}' in f
        ]
        if len(censor_files) > 0: 
            censor_file = os.listdir(f'{censor_path}/{censor_files[0]}')[0]
            censor_file = f"{censor_path}/{censor_files[0]}/{censor_file}"
            censor_times = pd.read_csv(censor_file).values.flatten()
            return censor_times
    except: return None

def get_mask(sub, ses, run):
    try:
        mask_files = [
            f 
            for f in os.listdir(mask_path)
            if f'SLC{sub:02d}' in f
            if f'ses-{ses}' in f
        ]
        if len(mask_files) > 0: 
            mask_run_files = [
                f
                for f in os.listdir(f'{mask_path}/{mask_files[0]}')
                if f'run_{run}' in f
            ]
            if len(mask_run_files) > 0:
                mask_file = os.listdir(f'{mask_path}/{mask_files[0]}/{mask_run_files[0]}')[0]
                mask_file = f'{mask_path}/{mask_files[0]}/{mask_run_files[0]}/{mask_file}'
                mask = image.load_img(mask_file)
                return mask
            else: return None
    except: return None

def get_data(sub, ses, run):
    try:
        data_files = [
            f 
            for f in os.listdir(data_path)
            if f'SLC{sub:02d}' in f
            if f'ses-{ses}' in f
            if f'run-{run}' in f
        ]
        if len(data_files) > 0: 
            data_file = os.listdir(f'{data_path}/{data_files[0]}')[0]
            data_file = f'{data_path}/{data_files[0]}/{data_file}'
            data = image.load_img(data_file)
            return data
    except: return None

# MAIN LOOP --------
for sub, ses in tqdm(product(args.subs, args.sess)):
    print(sub, ses)

    # stimulus----
    STIM = get_stim(sub, ses)

    # time series----
    keep_runs = []; remove_runs = []
    CENSOR = []; DATA = []
    cmask_data = np.ones(args.space_size)
    for run in np.arange(1, args.num_runs+1):
        if STIM[run-1] == ['*']: 
            remove_runs.append(run)
            continue
        
        censor_times = get_censor_times(sub, ses, run)
        mask = get_mask(sub, ses, run)
        data = get_data(sub, ses, run)

        if not (censor_times is None or mask is None or data is None):
            keep_runs.append(run)
            t = data.get_fdata()
            assert(t.shape[-1] == len(np.where(censor_times)[0]))
            ts = np.zeros((args.space_size+[args.num_times]))
            ts[:, :, :, np.where(censor_times)[0]] = t
            cmask_data *= mask.get_fdata()
            CENSOR.append(censor_times)
            DATA.append(ts)
        else:
            remove_runs.append(run)
            STIM[run-1] = ['*']
            
    for run in sorted(remove_runs, reverse=True):
        del STIM[run-1]
        
    # saving----
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
    
    # save space mask in a .nii.gx file: per sub and per ses
    image.new_img_like(mask, cmask_data, copy_header=True).to_filename(
        f'{REG_path}/sub-SLC{sub:02d}_ses-{ses}_desc-MASK.nii.gz'
    )
        
    # save DATA in a .1D file: per sub and per ses
    image.new_img_like(
        data, 
        np.nan_to_num(np.concatenate(DATA, axis=-1)),
        copy_header=True
    ).to_filename(
        f'{REG_path}/sub-SLC{sub:02d}_ses-{ses}_desc-INPUT.nii.gz'
    )
    
    # save CENSOR in a .txt file: per sub and per ses
    CENSOR = np.nan_to_num(np.hstack(CENSOR))
    np.savetxt(f'{REG_path}/sub-SLC{sub:02d}_ses-{ses}_desc-CENSOR.txt', CENSOR)


# In[4]:


# sub, ses = 1, 1
# mask = image.load_img(f'{REG_path}/sub-SLC{sub:02d}_ses-{ses}_desc-MASK.nii.gz').get_fdata()
# temp = image.load_img('/home/govindas/mouse_dataset/voxel/regression_analysis/Symmetric_N162_0.20_permuted.nii.gz').get_fdata()
# mask.shape, temp.shape

