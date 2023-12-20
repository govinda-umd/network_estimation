#!/bin/bash

# COMMAND TO RUN?
# bash 01-data-mouse_desc-run-svinet.sh \
# whole \
# yale_172 \
# ~/mouse_dataset/n162_parcellations/172_roi_labels.txt \
# 2 \
# 10

PARCELS=$1 #"whole" <<<<=======
ROIS_folder=$2 # hadi_1445, yale_172 <<<<=======

FILE_path=~/mouse_dataset/roi/${ROIS_folder}/func_nws_files/${PARCELS}

roi_labels=$3
NUM_ROIS=$(python -c "import numpy as np; print(len(np.loadtxt('${roi_labels}')))")

START_NUM_COMMS=$4
END_NUM_COMMS=$5

run_svinet() {
    for file in $(ls "${FILE_path}")
    do 
        echo "${file}"
        echo "----------"
        bash 01-data-mouse_desc-svinet.sh "${PARCELS}" "${ROIS_folder}" "${FILE_path}/${file}" "${NUM_ROIS}" "${NUM_COMMS}"
        echo "----------"
    done
}

for NUM_COMMS in $(seq ${START_NUM_COMMS} ${END_NUM_COMMS})
do
    echo NOW RUNNING FOR ${NUM_COMMS} COMMUNITIES 
    echo ========================================
    run_svinet
done