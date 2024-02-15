#!/bin/bash


# input fMRI paths
FILE_path=~/mouse_dataset/voxel/all_file_collections

# roi description
NUM_ROIS=$1
ROI_SIZE=$2
SYMM=$3
BRAIN_DIV=$4

DESC=type-functional_nrois-"${NUM_ROIS}"_size-"${ROI_SIZE}"_symm-"${SYMM}"_braindiv-"${BRAIN_DIV}"
PARCELS=~/mouse_dataset/allen_atlas_ccfv3/MouseConnectivity/parcels/"${DESC}"_desc-parcels.nii.gz

TS_FOLDER=~/mouse_dataset/roi/"${DESC}"/roi_timeseries
mkdir -p "${TS_FOLDER}"

for file in $(ls ~/mouse_dataset/voxel/all_file_collections)
do
    echo "${file}"
    bash 01a-desc-roi_timeseries.sh "${FILE_path}/${file}" "${PARCELS}" "${TS_FOLDER}"
    echo "-----------"
done