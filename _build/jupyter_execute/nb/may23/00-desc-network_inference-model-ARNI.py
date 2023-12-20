#!/usr/bin/env python
# coding: utf-8

# # May 15-16, 2023: Model-free inference of directed networks
# - paper: https://doi.org/10.1038/s41467-017-02288-4
# - code: https://github.com/networkinference/ARNI

# In[1]:


import csv
import os
import pickle
import random
import sys
from os.path import join as pjoin
import numpy as np
import scipy as sp 
from scipy.spatial.distance import pdist, cdist, squareform
from scipy import stats
import tvb
import networkx as nx
import copy
import matlab.engine
from itertools import product
from tqdm import tqdm

sys.path.append("/usr/local/MATLAB/R2022b/bin/matlab")

# main dirs
proj_dir = pjoin(os.environ['HOME'], 'network_estimation')
month_dir = f"{proj_dir}/nb/may23"
bdmodels_dir = f"{proj_dir}/helpers/bdmodels"
networks_dir = f"{proj_dir}/helpers/networks"
results_dir = f"{proj_dir}/results"

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams["errorbar.capsize"] = 0.5

import cmasher as cmr  # CITE ITS PAPER IN YOUR MANUSCRIPT

# folders
sys.path.insert(0, proj_dir)
import helpers.functions.network_utils as nw_utils
import helpers.functions.plotting_utils as plot_utils
import helpers.functions.time_series_simulation_utils as ts_sim_utils
import helpers.functions.fc_utils as fc_utils
import helpers.inference.ARNI as arni


# In[2]:


# network
W = sp.io.loadmat(f"{networks_dir}/networks_numrois_[5 5 5].mat")['networks'][0, :, :]

# time series
with open(f"{results_dir}/out_dicts_kuramoto.pkl", 'rb') as f:
    out_dicts = pickle.load(f)

# reconstruct
idx_subj, idx_sigma = 0, 0
out_dict = out_dicts[f"subj{idx_subj:02}"][f"sigma{idx_sigma:02}"][f"run{8:02}"]
X = out_dict['x'].T

W = W
MODEL, ORDER, BASIS = 'a', 15, 'polynomial'

reconstructions = []
for idx_node in tqdm(np.arange(X.shape[0])):
    reconst = arni.reconstruct(X, MODEL, ORDER, BASIS, idx_node, W)
    reconstructions.append(reconst) # llist, cost, FPR, TPR, AUC


# In[4]:


def get_inferred_network(W, reconstructions):
    W_ = np.zeros_like(W)
    for idx_node, reconst in enumerate(reconstructions):
        W_[idx_node, reconst[0]] = 1
    return W_


# In[5]:


W_ = get_inferred_network(W, reconstructions)


# In[6]:


W_

