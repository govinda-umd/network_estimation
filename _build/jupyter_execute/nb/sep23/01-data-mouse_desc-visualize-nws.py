#!/usr/bin/env python
# coding: utf-8

# # Sep 29, 2023; visualize overlapping networks

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


args.num_coms = 10


# In[4]:


TS_path = f'/home/govindas/mouse_dataset/voxel/voxel_timeseries_txt_files'
NW_path = f'/home/govindas/mouse_dataset/voxel/svinet'


# In[5]:


nw_files = os.listdir(NW_path)
nw_file = nw_files[0]

nw_groups = np.loadtxt(f'{NW_path}/{nw_file}/groups.txt')
nw_groups.shape


# In[6]:


ns = nw_file.split('_')
sub = [s for s in ns if f'SLC' in s][0].split('-')[-1]
ses = [s for s in ns if f'ses' in s][0].split('-')[-1]
run = [s for s in ns if f'run' in s][0].split('-')[-1]
task = [s for s in ns if f'task' in s][0].split('-')[1]
ts_file = f'sub-{sub}_ses-{ses}_run-{run}_task-{task}_desc-ts.txt'
ts = np.loadtxt(f'{TS_path}/{ts_file}')


# In[7]:


ts[:, 3:] = 0
ts[nw_groups[:, 1].astype(np.int), 3:3+args.num_coms] = nw_groups[:, 2:]


# In[8]:


fmt = ['%d' for _ in range(3)] + ['%.4f' for _ in range(args.num_coms)]
np.savetxt(f'/home/govindas/mouse_dataset/voxel/tmp/nw_groups_voxel.txt', ts[:, :3+args.num_coms], fmt=fmt)

