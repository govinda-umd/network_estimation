#!/usr/bin/env python
# coding: utf-8

# # Sep 26, 2023: collect data files per sub, ses, run

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


# In[3]:


censor_path = f'/home/govindas/mouse_dataset/voxel/frame_censoring_mask'
mask_path = f'/home/govindas/mouse_dataset/voxel/commonspace_mask'
data_path = f'/home/govindas/mouse_dataset/voxel/cleaned_timeseries'
COLLECT_path = f'/home/govindas/mouse_dataset/voxel/all_file_collections'
TS_path = f'/home/govindas/mouse_dataset/voxel/voxel_timeseries_txt_files'
ROI_TS_path = f'/home/govindas/mouse_dataset/roi/roi_timeseries_txt_files'


# In[4]:


def get_censor_file(sub, ses, run):
    try:
        censor_files = [
            f 
            for f in os.listdir(censor_path)
            if f'SLC{sub:02d}' in f
            if f'ses-{ses}' in f
            if f'run-{run}' in f
        ]
        censor_file = os.listdir(f'{censor_path}/{censor_files[0]}')[0]
        assert(censor_file.split('/')[-1].split('.')[-1] == 'csv')
        censor_file = f"{censor_path}/{censor_files[0]}/{censor_file}"
        return censor_file
    except: return None

def get_mask_file(sub, ses, run):
    try:
        mask_files = [
            f 
            for f in os.listdir(mask_path)
            if f'SLC{sub:02d}' in f
            if f'ses-{ses}' in f
        ]
        mask_run_files = [
            f
            for f in os.listdir(f'{mask_path}/{mask_files[0]}')
            if f'run_{run}' in f
        ]
        mask_file = os.listdir(f'{mask_path}/{mask_files[0]}/{mask_run_files[0]}')[0]
        assert(mask_file.split('/')[-1].split('.')[-1] == 'gz')
        mask_file = f'{mask_path}/{mask_files[0]}/{mask_run_files[0]}/{mask_file}'
        return mask_file
    except: return None

def get_data_file(sub, ses, run):
    try:
        data_files = [
            f 
            for f in os.listdir(data_path)
            if f'SLC{sub:02d}' in f
            if f'ses-{ses}' in f
            if f'run-{run}' in f
        ]
        data_file = os.listdir(f'{data_path}/{data_files[0]}')[0]
        assert(data_file.split('/')[-1].split('.')[-1] == 'gz')
        data_file = f'{data_path}/{data_files[0]}/{data_file}'
        return data_file
    except: return None


# In[5]:


for (sub, ses) in tqdm(product(args.subs, args.sess)):
        
    for run in np.arange(1,args.num_runs+1,2):
        censor_file = get_censor_file(sub, ses, run)
        mask_file = get_mask_file(sub, ses, run)
        data_file = get_data_file(sub, ses, run)
        
        if (censor_file is None or mask_file is None or data_file is None):
            continue
        
        task = [
            t 
            for t in censor_file.split('/')[-1].split('_') 
            if 'task' in t
        ][0].split('-')[-1]
        
        identity = f'sub-SLC{sub:02d}_ses-{ses}_run-{run}_task-{task}'
        ts_file = f'{TS_path}/{identity}_desc-ts.txt'
        roi_ts_file = f'{ROI_TS_path}/{identity}_desc-ts.txt'
        with open(
            f'{COLLECT_path}/{identity}_desc-files.txt', 
            'w', newline=''
        ) as f:
            wr = csv.writer(f, delimiter='\t')
            wr.writerows([
                [censor_file], [mask_file], 
                [data_file], [ts_file], 
                [roi_ts_file],
            ])

