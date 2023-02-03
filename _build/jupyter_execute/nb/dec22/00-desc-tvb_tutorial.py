#!/usr/bin/env python
# coding: utf-8

# # December 20, 2022: The Virtual Brain (TVB) tutorial

# In[1]:


import numpy as np
from tvb.simulator.lab import *
import tvb.simulator.lab as tsl


# In[2]:


# Create empty connectivity
wm = connectivity.Connectivity()

# First weights and distances
nor = 4
wm.motif_all_to_all(number_of_regions=nor)

# Centers, specify the number of regions, otherwise it'll use a default value.
wm.centres_spherical(number_of_regions=nor)

# By default, the new regions labels are numeric characters, ie [0, 1, ...]
wm.create_region_labels(mode='alphabetic')

# But custom region labels can be used
wm.region_labels = np.array('a1 a2 a3 a4'.split())
wm.configure()

# plot_matrix(wm.weights, connectivity=wm, binary_matrix=True)


# In[3]:


plot_matrix(wm.weights, connectivity=wm, binary_matrix=True)


# In[4]:


tsl.plot_connectivity(
    wm, 
    num='weights'
)


# In[5]:


wm = tsl.connectivity.Connectivity()
wm.number_of_regions = 5
wm.weights = np.random.rand(5, 5)
wm.centres_spherical(number_of_regions=5)
wm.region_labels = np.array([0, 1, 2, 3, 4])
wm.tract_lengths = 100 * wm.weights
wm.configure()

tsl.plot_connectivity(
    wm,
)


# In[6]:


wm = tsl.connectivity.Connectivity.from_file()
wm

