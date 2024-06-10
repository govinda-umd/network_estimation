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

RECONST_method=${6}

# sbm descriptors
SBMs=$7 # a/d/h/o
DCs=$8 # 1/0
WAIT=${9} #1000/3000/5000
SEED=${10} #100

# num jobs in parallel
MAX_PROCS=${11} #5/10/15

ALL_GRAPHS=${HOME}/mouse_dataset/roi_results/${DESC}/seswise/reconstructed_graph/${RECONST_method}/all_graphs.txt

parallel --max-procs ${MAX_PROCS} \
python 06a-desc-fit-sbm.py \
::: ${DESC} \
::: `cat ${ALL_GRAPHS}` \
::: ${SBMs} ::: ${DCs} \
::: ${WAIT} ::: ${SEED}