#!/bin/bash

# source ~/.bashrc

TYPE=${1}
SIZE=${2}
SYMM=${3}
BRAIN_DIV=${4}
NROIS=${5}

PARC_DESC=type-${TYPE}_size-${SIZE}_symm-${SYMM}_braindiv-${BRAIN_DIV}_nrois-${NROIS}

GRAPH_DEF="constructed"
GRAPH_METHOD=${6} #pearson-corr
THRESHOLDING=${7} #positive/absolute
EDGE_DEF="binary"
EDGE_DENSITY=${8}
LAYER_DEF="individual"
DATA_UNIT="ses"

WAIT="12" #"24000"
MCMC_LEN="100" #"40000"
Bs="1 41 82 122 162"
Bs=($Bs)
SEED="100"

for B in "${Bs[@]}"
do
    sbatch 02a-graph-individual-binary_model-dcoSBM_desc-fit-sbm.sh \
        ${TYPE} ${SIZE} ${SYMM} ${BRAIN_DIV} ${NROIS} \
        ${GRAPH_DEF} ${GRAPH_METHOD} ${THRESHOLDING} ${EDGE_DENSITY} ${DATA_UNIT} \
        ${WAIT} ${MCMC_LEN} ${B} ${SEED}
done

# bash 02a-graph-individual-binary_model-aSBM_desc-run-fit-sbm.sh \
# spatial 225 True whl 162 pearson-corr positive 10