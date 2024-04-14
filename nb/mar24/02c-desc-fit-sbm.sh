#!/bin/bash

# COMMAND TO RUN IN BASH
# bash 02c-desc-fit-sbm.sh spatial 225 True whl 162 "a d h o" "1 0" 100 100

# roi descriptors
TYPE=$1
ROI_SIZE=$2
SYMM=$3
BRAIN_DIV=$4
NUM_ROIS=$5

DESC=type-"${TYPE}"_size-"${ROI_SIZE}"_symm-"${SYMM}"_braindiv-"${BRAIN_DIV}"_nrois-"${NUM_ROIS}"

# sbm descriptors
SBMs=$6
DCs=$7
WAIT=${8}
SEED=${9}

# num jobs in parallel
MAX_PROCS=${10}

ALL_GRAPHS=${HOME}/mouse_dataset/roi_results/${DESC}/all_graphs.txt


parallel --max-procs ${MAX_PROCS} \
python 02c-desc-fit-sbm.py \
::: ${DESC} \
::: `cat ${ALL_GRAPHS}` \
::: ${SBMs} ::: ${DCs} \
::: ${WAIT} ::: ${SEED}
