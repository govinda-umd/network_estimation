#!/usr/bin/env python
# coding: utf-8

# # Sep 27, 2023: functional connectivity

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
from scipy import sparse

# nilearn
from nilearn import image

# networkx
import networkx as nx 

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
warnings.filterwarnings("ignore") #, category=UserWarning)


# In[2]:


class ARGS():
    pass

args = ARGS()


# In[3]:


TS_path = f'/home/govindas/mouse_dataset/voxel/voxel_timeseries_txt_files'
NW_EDGES_path = f'/home/govindas/mouse_dataset/voxel/nw_edges'
ts_files = os.listdir(TS_path)


# In[4]:


for ts_file in tqdm(ts_files):
    ts = np.loadtxt(f'{TS_path}/{ts_file}').T[3:, :] # time x vox
    # FC
    R = np.corrcoef(ts, rowvar=False)
    R = np.nan_to_num(R)
    thresh = np.nanpercentile(np.abs(R).flatten(), q=95,)
    R *= R > thresh
    # print(R.shape)
    R = np.triu(R, k=1)
    # edge list
    E = np.stack(np.where(R), axis=-1)
    edges_file = ts_file.split('_')
    edges_file[-1] = 'desc-nw-edges.txt'
    edges_file = '_'.join(edges_file)
    with open(f'{NW_EDGES_path}/{edges_file}', 'w', newline='') as f:
        wr = csv.writer(f, delimiter='\t')
        wr.writerows(E)
    # clear memory
    del ts
    del R
    del E

