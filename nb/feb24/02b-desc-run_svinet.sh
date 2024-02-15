#!/bin/bash

# roi description
NUM_ROIS=$1
ROI_SIZE=$2
SYMM=$3
BRAIN_DIV=$4

DESC=type-functional_nrois-"${NUM_ROIS}"_size-"${ROI_SIZE}"_symm-"${SYMM}"_braindiv-"${BRAIN_DIV}"
FC_FOLDER=~/mouse_dataset/roi/"${DESC}"/func_nws
SVINET_FOLDER=~/mouse_dataset/roi/"${DESC}"/svinets
mkdir -p "${SVINET_FOLDER}"

START_NUM_COMMS=$5
END_NUM_COMMS=$6

START_SEED=$7
END_SEED=$8

run_svinet(){
    for file in $(ls "${FC_FOLDER}")
    do
        echo "${file}"
        echo "========="
        for seed in $(seq ${START_SEED} ${END_SEED})
        do
            echo "${seed}"
            bash 02a-desc-svinet.sh "${FC_FOLDER}/${file}" "${SVINET_FOLDER}" \
            "${NUM_ROIS}" "${NUM_COMMS}" "${seed}"
            echo "---------"
        done
    done
}

for NUM_COMMS in $(seq ${START_NUM_COMMS} ${END_NUM_COMMS})
do
    echo NOW RUNNING FOR ${NUM_COMMS} COMMUNITIES 
    echo ========================================
    run_svinet &
done