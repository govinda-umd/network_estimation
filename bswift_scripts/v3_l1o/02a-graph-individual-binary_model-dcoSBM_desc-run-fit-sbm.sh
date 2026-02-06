#!/bin/bash

# source ~/.bashrc

SOURCE=${1} #allen
SPACE=${2} #ccfv2
BRAIN_DIV=${3} #whl
NROIS=${4} # 172
RES=${5} # resolution 200

PARC_DESC=source-${SOURCE}_space-${SPACE}_braindiv-${BRAIN_DIV}_nrois-${NROIS}_res-${RES}

GRAPH_DEF="constructed"
GRAPH_METHOD=${6} #pearson
THRESHOLDING=${7} #signed/unsigned
EDGE_DEF="binary"
EDGE_DENSITY=${8} #10,20
LAYER_DEF="individual"
DATA_UNIT=${9} #'grp-boot'

WAIT="24000"
MCMC_LEN="3000"
SEGMENT_LEN="300"
Bs="1 112 224 335 446"
Bs=($Bs)
GAMMA="2.0"
SEED="100"

for B in "${Bs[@]}"
do
    sbatch 02a-graph-individual-binary_model-dcoSBM_desc-fit-sbm.sh \
        ${SOURCE} ${SPACE} ${BRAIN_DIV} ${NROIS} ${RES} \
        ${GRAPH_DEF} ${GRAPH_METHOD} ${THRESHOLDING} ${EDGE_DENSITY} ${DATA_UNIT} \
        ${WAIT} ${MCMC_LEN} ${SEGMENT_LEN} ${B} ${GAMMA} ${SEED}
done

# bash 02a-graph-individual-binary_model-dcoSBM_desc-run-fit-sbm.sh \
# allen ccfv2 whl 172 200 pearson signed 20 sub