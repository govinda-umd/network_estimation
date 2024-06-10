#!/bin/bash

# COMMAND TO RUN IN BASH
# bash 02c-desc-fit-sbm.sh spatial 225 True whl 162 "a d h o" "1 0" 100 100

# roi descriptors
TYPE=$1 #spatial
ROI_SIZE=$2 #225
SYMM=$3 #True
BRAIN_DIV=$4 #whl
NUM_ROIS=$5 #162

DESC=type-"${TYPE}"_size-"${ROI_SIZE}"_symm-"${SYMM}"_braindiv-"${BRAIN_DIV}"_nrois-"${NUM_ROIS}"

# sbm descriptors
SBMs=$6 # a/d/h/o
DCs=$7 # 1/0
WAIT=${8} #1000/3000/5000
SEED=${9} #100

# num jobs in parallel
MAX_PROCS=${10} #5/10/15

ALL_GRAPHS=${HOME}/mouse_dataset/roi_results/${DESC}/all_graphs.txt


parallel --max-procs ${MAX_PROCS} \
python 02c-desc-fit-sbm.py \
::: ${DESC} \
::: `cat ${ALL_GRAPHS}` \
::: ${SBMs} ::: ${DCs} \
::: ${WAIT} ::: ${SEED}
