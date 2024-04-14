#!/bin/bash


# input fMRI paths
FILE_path=~/mouse_dataset/voxel/all_file_collections

# roi description
TYPE=$1
ROI_SIZE=$2
SYMM=$3
BRAIN_DIV=$4
NUM_ROIS=$5

DESC=type-"${TYPE}"_size-"${ROI_SIZE}"_symm-"${SYMM}"_braindiv-"${BRAIN_DIV}"_nrois-"${NUM_ROIS}"
PARCELS=~/mouse_dataset/parcels/"${DESC}"_desc-parcels.nii.gz

TS_FOLDER=~/mouse_dataset/roi/"${DESC}"/roi_timeseries
mkdir -p "${TS_FOLDER}"

for file in $(ls ~/mouse_dataset/voxel/all_file_collections)
do
    echo "${file}"
    bash 01a-desc-roi_timeseries.sh "${FILE_path}/${file}" "${PARCELS}" "${TS_FOLDER}"
    echo "-----------"
done