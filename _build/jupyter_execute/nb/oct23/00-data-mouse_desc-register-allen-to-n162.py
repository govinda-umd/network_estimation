#!/usr/bin/env python
# coding: utf-8

# # Oct 4-11, 2023: Allen Atlas CCFv3: register to N162

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
from nipype.interfaces import afni 

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

args.out_path = (
    f'/home/govindas/mouse_dataset/allen_atlas_ccfv3' 
    f'/hadi/parcellation'
)


# ## template registration

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


STree


# In[5]:


STree[STree.id==595].name


# In[6]:


def to_nifti(args, img=AVGT, file_name='allen'):
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
    img_file = (
        f'{args.out_path}'
        f'/warped_on_n162/{file_name}.nii.gz'
    )
    return ants_img, img_file


# In[7]:


ants_img_allen, img_allen_file = to_nifti(args, AVGT, 'allen')
ants_img_ano, img_ano_file = to_nifti(args, ANO, 'allen_ano')


# In[8]:


img_n162_file = f'/home/govindas/mouse_dataset/gabe_symmetric_N162/Symmetric_N162_0.10_RAS.nii.gz'
ants_img_n162 = ants.image_read(img_n162_file)


# In[9]:


tx = ants.registration(
    fixed=ants_img_n162,
    moving=ants_img_allen,
    type_of_transform=('SyN'),
)


# In[10]:


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
ants_img_allen_warped.to_filename(f'{args.out_path}/warped_on_n162/allen_warped.nii.gz')
ants_img_ano_warped.to_filename(f'{args.out_path}/warped_on_n162/ano_warped.nii.gz')


# In[11]:


isocortex_id = STree[STree.name == 'Isocortex'].id.values[0]
ISO, metaISO = mcc.get_structure_mask(isocortex_id)
ISO = ISO.astype(np.uint32)
ants_img_allen_iso, img_allen_iso_file = to_nifti(args, ISO, 'allen_iso')
ants_img_iso_warped = ants.apply_transforms(
    fixed=ants_img_n162,
    moving=ants_img_allen_iso,
    transformlist=tx['fwdtransforms'],
    interpolator='genericLabel',
)
ants_img_iso_warped.to_filename(f'{args.out_path}/warped_on_n162/allen_iso_warped.nii.gz')


# ## parcellation

# In[12]:


regions_df = pd.read_pickle(f'{args.out_path}/regions.df')


# In[13]:


acros = regions_df.acro.unique()
print(acros)


# In[14]:


def create_parcels(args, parcels, rois_df, name):
    new_parcels = np.zeros_like(parcels)
    new_parcels = new_parcels.astype(np.int32)

    for idx, row in rois_df.iterrows():
        new_parcels += (parcels == row.id) * row.id
    new_parcels = new_parcels.astype(np.int32)

    new_parcels_img, new_parcels_file = to_nifti(args, new_parcels, name)
    new_parcels_img_warped = ants.apply_transforms(
        fixed=ants_img_n162,
        moving=new_parcels_img,
        transformlist=tx['fwdtransforms'],
        interpolator='genericLabel',
    )
    print(np.unique(new_parcels).shape)
    return new_parcels, new_parcels_img_warped


# In[15]:


regions_df[~regions_df.layer.isin([0, 2])].shape


# In[16]:


parcels = np.load(f'{args.out_path}/brain_100um.npy')
parcels = parcels.astype(np.int32)

# removing rois in layers 0, 2 of isocortex
rois_df = regions_df[~regions_df.layer.isin([0, 2])]
whole_parcels, whole_parcels_img_warped = create_parcels(args, parcels, rois_df, 'whole_parcels')

rois_df = regions_df[regions_df.acro == 'Isocortex'][regions_df.layer == 1]
iso_parcels, iso_parcels_img_warped = create_parcels(args, parcels, rois_df, 'iso_parcels')

rois_df = regions_df[regions_df.acro.isin(['Isocortex', 'OLF'])][~regions_df.layer.isin([0, 2])]
iso_olf_parcels, iso_olf_parcels_img_warped = create_parcels(args, parcels, rois_df, 'iso_olf_parcels')

rois_df = regions_df[regions_df.acro == 'OLF']
olf_parcels, olf_parcels_img_warped = create_parcels(args, parcels, rois_df, 'olf_parcels')

rois_df = regions_df[~regions_df.acro.isin(['Isocortex', 'OLF'])]
rest_parcels, rest_parcels_img_warped = create_parcels(args, parcels, rois_df, 'non_iso_olf_parcels')


# ## resampling

# In[17]:


# common brain mask (across subs)
all_files_path = f'/home/govindas/mouse_dataset/voxel/all_file_collections'
all_files = os.listdir(all_files_path)

# cmask : common brain mask
for idx, files in tqdm(enumerate(all_files[:])):
    if idx == 0:
        with open(f'{all_files_path}/{files}', 'r') as f:
            ants_cmask = ants.image_read(f.readlines()[1][:-1])
        cmask = ants_cmask.numpy()
    else:
        with open(f'{all_files_path}/{files}', 'r') as f:
            cmask *= ants.image_read(f.readlines()[1][:-1]).numpy()
ants_cmask = ants_cmask.new_image_like(cmask)
ants_cmask.to_filename(
    f'/home/govindas/mouse_dataset/voxel/common_brain_mask.nii.gz'
)


# In[18]:


def resample_to_common_mask(args, cmask_img, parcels_img, name):
    parcels_img_warped = ants.resample_image_to_target(
        image=parcels_img,
        target=cmask_img,
        interp_type='genericLabel',
    )
    parcels_img_warped = parcels_img_warped.new_image_like(
        data=parcels_img_warped.numpy() * cmask_img.numpy()
    )
    fname = f'{args.out_path}/warped_on_n162/{name}_warped_cm.nii.gz'
    print(fname)
    parcels_img_warped.to_filename(
        fname
    )
    return fname


# In[19]:


whole_parcels_img_warped_cm = resample_to_common_mask(
    args, cmask_img=ants_cmask, parcels_img=whole_parcels_img_warped, name='whole_parcels')

iso_parcels_img_warped_cm = resample_to_common_mask(
    args, cmask_img=ants_cmask, parcels_img=iso_parcels_img_warped, name='iso_parcels')

iso_olf_parcels_img_warped_cm = resample_to_common_mask(
    args, cmask_img=ants_cmask, parcels_img=iso_olf_parcels_img_warped, name='iso_olf_parcels')

olf_parcels_img_warped_cm = resample_to_common_mask(
    args, cmask_img=ants_cmask, parcels_img=olf_parcels_img_warped, name='olf_parcels')

non_iso_olf_parcels_img_warped_cm = resample_to_common_mask(
    args, cmask_img=ants_cmask, parcels_img=rest_parcels_img_warped, name='non_iso_olf_parcels')


# In[20]:


def roi_labels(args, mask_file, name):
    cmd = (
        f'3dROIstats -overwrite '
        f'-quiet '
        f'-mask {mask_file} '
        f'{mask_file} > {args.out_path}/warped_on_n162/{name}_roi_labels.txt'
    )
    os.system(cmd)
    return None

roi_labels(args, whole_parcels_img_warped_cm, 'whole')
roi_labels(args, iso_parcels_img_warped_cm, 'iso')
roi_labels(args, iso_olf_parcels_img_warped_cm, 'iso_olf')
roi_labels(args, olf_parcels_img_warped_cm, 'olf')
roi_labels(args, non_iso_olf_parcels_img_warped_cm, 'non_iso_olf')

