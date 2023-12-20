#!/usr/bin/env python
# coding: utf-8

# # Sep 3, 2023: Emoprox2 dataset: ICA

# In[1]:


import csv
import os
import numpy as np
import pandas as pd
import scipy as sp 
import pickle 
from sklearn.decomposition import FastICA, PCA
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


class ARGS(): pass
args = ARGS()

args.SEED = 100
np.random.seed(args.SEED)


# In[3]:


emoprox_dir = pjoin(os.environ['HOME'], 'emoprox_inventory_govinda/emoprox_extract_timeseries')

# ROI atlas/parcellation
atlas_df = pd.read_csv(f"{emoprox_dir}/masks/README_MAX_ROIs_final_gm_85.txt", delimiter='\t')
# display(atlas_df)

roi = 'dorsal Insula'
roi_idx = np.where(atlas_df['Hemi'].str.contains('R') * atlas_df['ROI'].str.contains(roi))[0]
display(atlas_df.iloc[roi_idx])

# data
data_df = pd.read_pickle(f"{emoprox_dir}/dataframes/MAX_ROIs.pkl")

data_df['proximity'] = data_df['proximity'].apply(lambda prox: sp.stats.zscore(prox))


# In[4]:


# mean time series across subjects
rids = []
blocks = []
times = []
tss = []
proxs = []
censors = []
for rid, block in list(product(data_df['rid'].unique(), data_df['block'].unique())):
    # print(rid, block)
    rids.append(rid)
    blocks.append(block)
    df_ = data_df[data_df['rid'] == rid][data_df['block'] == block]
    lmin = np.min(df_['timeseries'].apply(lambda x: x.shape[0]).values)
    times.append(df_['time'].iloc[0][:lmin])
    tss.append(df_['timeseries'].apply(lambda x: x[:lmin, ...]).mean())
    proxs.append(df_['proximity'].apply(lambda x: x[:lmin, ...]).mean())
    censors.append(df_['censor'].values[0][:lmin, 0])

df = pd.DataFrame(
    data={
        'rid': rids,
        'block': blocks,
        'time': times,
        'timeseries': tss,
        'proximity': proxs,
        'censor':censors,
        },
)
df


# ## spatial ica

# In[5]:


args.n_comps = 10
ica = FastICA(
    n_components=args.n_comps, 
    whiten='unit-variance',
    max_iter=1000,
)
X = df.iloc[0]['timeseries']
t = df.iloc[0]['time']
prox = df.iloc[0]['proximity']
censor = df.iloc[0]['censor']
S = ica.fit_transform(X)
A = ica.mixing_
print(S.shape, A.shape, np.linalg.matrix_rank(A))
X_ = ica.inverse_transform(S)
# np.allclose(X, X_)


# In[6]:


R = []; P = []
for i in range(S.shape[1]):
   res = sp.stats.pearsonr(S[:, i], prox)
   R.append(res.statistic)
   P.append(res.pvalue)
R = np.array(R)
P = np.array(P)


# In[7]:


fig, axs = plt.subplots(
    nrows=args.n_comps, ncols=1,
    figsize=(15, 3.5*args.n_comps),
    dpi=75
)
# prox_ = deepcopy(prox)
prox_ = np.roll(prox, 1)
for idx in range(args.n_comps):
    ax = axs[idx]
    ax.plot(t, S[:, idx], linewidth=3, label=f'src{idx}')
    ax.plot(t, prox_, linewidth=3, label='prox')
    ax.plot(t, 1-censor, linewidth=2, c='k', label='cnsr')
    ax.text(75, -2.0, f"r={R[idx]:.2f}, p={P[idx]:.3f}")
    ax.legend()
    ax.grid(True)


# source 7 follows shock onset. 
# source 5 follows proximity stimulus.

# In[8]:


R = np.corrcoef(A.T)
fig, axs = plt.subplots(1, 1, figsize=(8, 6))
ax = axs
sns.heatmap(R, vmin=-1, vmax=1, square=True, cmap=cmr.iceburn)


# ## pca

# In[9]:


pca = PCA(n_components=args.n_comps)
Y = pca.fit_transform(X)
U = pca.components_
print(U.shape, Y.shape)
X_ = pca.inverse_transform(Y)
X_.shape


# In[10]:


fig, axs = plt.subplots(1, 2, figsize=(11, 8))
ax = axs[0]
sns.heatmap(X, ax=ax)
ax = axs[1]
sns.heatmap(X_, ax=ax)

np.cumsum(pca.explained_variance_ratio_)


# In[11]:


R = np.corrcoef(U)
fig, axs = plt.subplots(1, 1, figsize=(8, 6))
ax = axs
sns.heatmap(R, vmin=-1, vmax=1, square=True, cmap=cmr.iceburn)


# ## temporal ica

# In[12]:


args.n_comps = 10
ica = FastICA(
    n_components=args.n_comps, 
    whiten='unit-variance',
    max_iter=1000,
)
X = df.iloc[0]['timeseries']
t = df.iloc[0]['time']
prox = df.iloc[0]['proximity']
censor = df.iloc[0]['censor']
S = ica.fit_transform(X.T)
A = ica.mixing_
print(S.shape, A.shape, np.linalg.matrix_rank(A))
X_ = ica.inverse_transform(S)
# np.allclose(X, X_)
A_ = sp.stats.zscore(A)


# In[13]:


R = []; P = []
for i in range(S.shape[1]):
   res = sp.stats.pearsonr(A_[:, i], prox)
   R.append(res.statistic)
   P.append(res.pvalue)
R = np.array(R)
P = np.array(P)


# In[14]:


fig, axs = plt.subplots(
    nrows=args.n_comps, ncols=1,
    figsize=(15, 3.5*args.n_comps),
    dpi=75
)
prox_ = deepcopy(prox)
# prox_ = np.roll(prox, 1)
for idx in range(args.n_comps):
    ax = axs[idx]
    ax.plot(t, A_[:, idx], linewidth=3, label=f'src{idx}')
    ax.plot(t, prox_, linewidth=3, label='prox')
    ax.plot(t, 1-censor, linewidth=2, c='k', label='cnsr')
    ax.text(75, -2.0, f"r={R[idx]:.2f}, p={P[idx]:.3f}")
    ax.legend()
    ax.grid(True)

