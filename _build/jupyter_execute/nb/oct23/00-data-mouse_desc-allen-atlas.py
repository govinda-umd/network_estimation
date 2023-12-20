#!/usr/bin/env python
# coding: utf-8

# # Oct 4, 2023: Allen Atlas CCFv3

# In[1]:


import csv
import os
import sys
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

# nii imaging
from allensdk.core.mouse_connectivity_cache import (
    MouseConnectivityCache,
    MouseConnectivityApi
)
import nrrd
import ants

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

# user files
path = f'/home/govindas/hadivafaii/Ca-fMRI'
sys.path.insert(0, path)
# from register import register

# ignore user warnings
import warnings
warnings.filterwarnings("ignore") #, category=UserWarning)


# In[2]:


class ARGS():
    pass

args = ARGS()


# In[3]:


atlas_path = f'/home/govindas/mouse_dataset/allen_atlas_ccfv3'
mcc_path = f'{atlas_path}/MouseConnectivity'
mcc = MouseConnectivityCache(
    resolution=100,
    manifest_file=f'{mcc_path}/manifest.json',
    ccf_version=MouseConnectivityApi.CCF_2017,
)
AVGT, metaAVGT = mcc.get_template_volume()
ANO, metaANO = mcc.get_annotation_volume()
AVGT = AVGT.astype(np.float32)
ANO = ANO.astype(np.uint32)
print(AVGT.shape, ANO.shape)

STree = pd.DataFrame(mcc.get_structure_tree().nodes()) 


# In[4]:


def save_to_nifti(img=AVGT, file_name='allen'):
    img = img.transpose(2, 0, 1)
    img = img[:,:,::-1]
    img = np.pad(
        img, 
        pad_width=((2, 2), (4, 24), (8, 2)), 
        mode='constant',
        constant_values=((0, 0), (0, 0), (0, 0))
        )
    print(img.dtype, img.shape)
    ndims = len(img.shape)
    ants_img = ants.from_numpy(
        data=img.astype(np.float32), 
        origin=[6.4, -13.2, -7.8],
        spacing=[0.1]*ndims,
    )
    img_file = f'/home/govindas/mouse_dataset/voxel/tmp/{file_name}.nii.gz'
    ants_img.to_filename(img_file)
    return ants_img, img_file


# In[5]:


ants_img_allen, img_allen_file = save_to_nifti(AVGT, 'allen')
ants_img_ano, img_ano_file = save_to_nifti(ANO, 'allen_ano')


# In[6]:


img_n162_file = f'/home/govindas/mouse_dataset/gabe_symmetric_N162/Symmetric_N162_0.10_RAS.nii.gz'
ants_img_n162 = ants.image_read(img_n162_file)
img_n162 = ants_img_n162.numpy()

img_n162_file = f'/home/govindas/mouse_dataset/voxel/tmp/n162.nii.gz'
ants_img_n162.to_filename(img_n162_file)


# In[7]:


tx = ants.registration(
    fixed=ants_img_n162,
    moving=ants_img_allen,
    type_of_transform=('SyN'),
)


# In[8]:


ants_img_allen_warped = ants.apply_transforms(
    fixed=ants_img_n162,
    moving=ants_img_allen,
    transformlist=tx['fwdtransforms'],
    interpolator='genericLabel',
)
ants_img_ano_warped = ants.apply_transforms(
    fixed=ants_img_n162,
    moving=ants_img_ano,
    transformlist=tx['fwdtransforms'],
    interpolator='genericLabel',
)
ants_img_allen_warped.to_filename(f'/home/govindas/mouse_dataset/voxel/tmp/allen_warped.nii.gz')
ants_img_ano_warped.to_filename(f'/home/govindas/mouse_dataset/voxel/tmp/ano_warped.nii.gz')


# In[9]:


isocortex_id = STree[STree.name == 'Isocortex'].id.values[0]
ISO, metaISO = mcc.get_structure_mask(isocortex_id)
ISO = ISO.astype(np.uint32)
ants_img_allen_iso, img_allen_iso_file = save_to_nifti(ISO, 'allen_iso')
ants_img_iso_warped = ants.apply_transforms(
    fixed=ants_img_n162,
    moving=ants_img_allen_iso,
    transformlist=tx['fwdtransforms'],
    interpolator='genericLabel',
)
ants_img_iso_warped.to_filename(f'/home/govindas/mouse_dataset/voxel/tmp/allen_iso_warped.nii.gz')

