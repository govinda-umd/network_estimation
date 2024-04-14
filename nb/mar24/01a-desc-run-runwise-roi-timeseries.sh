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

TS_FOLDER=~/mouse_dataset/roi_results/"${DESC}"/roi_timeseries
mkdir -p "${TS_FOLDER}"

# for file in $(ls ${FILE_path})
# do
#     echo "${file}"
#     bash 01a-desc-runwise-roi-timeseries.sh "${FILE_path}/${file}" "${PARCELS}" "${TS_FOLDER}"
#     echo "-----------"
# done


parallel --max-procs 10 \
bash 01a-desc-runwise-roi-timeseries.sh ::: ${FILE_path} ::: `ls ${FILE_path}` ::: ${PARCELS} ::: ${TS_FOLDER}