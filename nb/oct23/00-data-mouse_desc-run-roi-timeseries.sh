#!/bin/bash

# COMMAND TO RUN?
# bash 00-data-mouse_desc-run-roi-timeseries.sh \
# whole \
# ~/mouse_dataset/n162_parcellations/56_parcels_RAS_cm.nii.gz \
# yale_56

FILE_path=~/mouse_dataset/voxel/all_file_collections
PARCELs=$1 #"whole" #"iso" #"sub" <<<<=======
ROIS=$2 #<<<<=======
ROIS_folder=$3 #<<<<=======
# "~/mouse_dataset/n162_parcellations/172_parcels_RAS_cm.nii.gz"
# "~/mouse_dataset/allen_atlas_ccfv3/hadi/parcellation/warped_on_n162/${PARCELs}_parcels_warped_cm.nii.gz"

for file in $(ls ~/mouse_dataset/voxel/all_file_collections)
do 
    echo "${file}"
    # echo "----------"
    bash 00-data-mouse_desc-roi-timeseries.sh "${FILE_path}/${file}" "${PARCELs}" "${ROIS}" "${ROIS_folder}"
    echo "----------"
done