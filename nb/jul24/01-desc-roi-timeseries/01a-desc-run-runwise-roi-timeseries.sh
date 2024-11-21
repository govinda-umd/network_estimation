#!/bin/bash

# input fMRI paths
FILE_path=~/mouse_dataset/voxel/all_file_collections

# PARC_DESC=type-"${TYPE}"_size-"${ROI_SIZE}"_symm-"${SYMM}"_braindiv-"${BRAIN_DIV}"_nrois-"${NUM_ROIS}"
PARC_DESC=${1}
PARCELS=~/mouse_dataset/parcels/"${PARC_DESC}"_desc-parcels.nii.gz

# TS_FOLDER=~/mouse_dataset/roi_results_v2/"${PARC_DESC}"/runwise_timeseries
TS_FOLDER=${2}
mkdir -p "${TS_FOLDER}"

parallel --max-procs 10 \
bash 01a-desc-runwise-roi-timeseries.sh \
::: ${FILE_path} \
::: `ls ${FILE_path}` \
::: ${PARCELS} \
::: ${TS_FOLDER}