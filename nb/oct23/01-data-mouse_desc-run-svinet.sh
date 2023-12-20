#!/bin/bash

PARCELS=$1 # "whole" <<<<====
START_NUM_COMMS=$2
END_NUM_COMMS=$3


FILE_path=~/mouse_dataset/roi/func_nws_files/${PARCELS}

roi_labels=~/mouse_dataset/allen_atlas_ccfv3/hadi/parcellation/warped_on_n162/${PARCELS}_roi_labels.txt
NUM_ROIS=$(python -c "import numpy as np; print(len(np.loadtxt('${roi_labels}')))")


run_svinet() {
    for file in $(ls "${FILE_path}")
    do 
        echo "${file}"
        echo "----------"
        bash 01-data-mouse_desc-svinet.sh "${PARCELS}" "${FILE_path}/${file}" "${NUM_ROIS}" "${NUM_COMMS}"
        echo "----------"
    done
}

for NUM_COMMS in $(seq ${START_NUM_COMMS} ${END_NUM_COMMS})
do
    echo NOW RUNNING FOR ${NUM_COMMS} COMMUNITIES 
    echo ========================================
    run_svinet
done