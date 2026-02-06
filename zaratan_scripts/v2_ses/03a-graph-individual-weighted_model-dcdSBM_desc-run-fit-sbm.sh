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
EDGE_DEF="weighted"
EDGE_DENSITY=${8}
LAYER_DEF="individual"
DATA_UNIT="ses"

WAIT="24000"
MCMC_LEN="40000"
Bs="1 41 82 122 162"
Bs=($Bs)
TRANSFORM=${9} # arctanh
SEED="100"

for B in "${Bs[@]}"
do
    sbatch 03a-graph-individual-weighted_model-dcdSBM_desc-fit-sbm.sh \
        ${TYPE} ${SIZE} ${SYMM} ${BRAIN_DIV} ${NROIS} \
        ${GRAPH_DEF} ${GRAPH_METHOD} ${THRESHOLDING} ${EDGE_DENSITY} ${DATA_UNIT} \
        ${WAIT} ${MCMC_LEN} ${B} ${TRANSFORM} ${SEED}
done

# bash 03a-graph-individual-weighted_model-aSBM_desc-run-fit-sbm.sh \
# spatial 225 True whl 162 pearson-corr absolute 10 arctanh