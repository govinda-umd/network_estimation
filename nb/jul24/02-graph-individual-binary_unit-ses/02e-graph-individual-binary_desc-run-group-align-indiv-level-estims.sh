#!/bin/bash

TYPE=${1}
SIZE=${2}
SYMM=${3}
BRAIN_DIV=${4}
NROIS=${5}

PARC_DESC=type-${TYPE}_size-${SIZE}_symm-${SYMM}_braindiv-${BRAIN_DIV}_nrois-${NROIS}

GRAPH_DEF=${6} # constructed
GRAPH_METHOD=${7} # pearson-corr
THRESHOLDING=${8} # positive/absolute
EDGE_DEF="binary"
EDGE_DENSITY=${9} # 10, 15, 20, 25
LAYER_DEF="individual"
DATA_UNIT=${10} # ses, sub

SBM=${11} #"a" "d" "h"
DC=${12} #"True" # deg-corr

SEED=${13}

python 02e-graph-individual-binary_desc-group-align-indiv-level-estims.py \
    ${TYPE} ${SIZE} ${SYMM} ${BRAIN_DIV} ${NROIS} \
    ${GRAPH_DEF} ${GRAPH_METHOD} ${THRESHOLDING} ${EDGE_DEF} ${EDGE_DENSITY} ${LAYER_DEF} ${DATA_UNIT} \
    ${DC} ${SBM} ${SEED}

# COMMAND TO RUN ON BASH
# parallel --max-procs 5 \
# bash 02e-graph-individual-binary_desc-run-group-align-indiv-level-estims.sh \
# ::: spatial ::: 225 ::: True ::: whl ::: 162 ::: constructed ::: pearson-corr ::: positive ::: 10 ::: ses ::: a d h ::: True False ::: 100 