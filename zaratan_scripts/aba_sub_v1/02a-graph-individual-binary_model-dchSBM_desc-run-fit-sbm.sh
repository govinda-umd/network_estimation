#!/bin/bash

# source ~/.bashrc

PARC_DESC="NEWMAX_ROIs_final_gm_100_2mm" #"ABA_ROIs_final_gm_36"
ANALYSIS="trial-end"
GRAPH_DEF="constructed"
GRAPH_METHOD="pearson"
THRESHOLDING=${1} #signed/unsigned
EDGE_DEF="binary"
EDGE_DENSITY=${2} #10,20
LAYER_DEF="individual"
DATA_UNIT="sub"
COND=${3} #highT, highR, lowT, lowR

WAIT="1200"
MCMC_LEN="100"
Bs="1 10 19 28 37"
Bs=($Bs)
GAMMA="2.0"
SEED="100"

for B in "${Bs[@]}"
do
    sbatch 02a-graph-individual-binary_model-dchSBM_desc-fit-sbm.sh \
        ${PARC_DESC} ${ANALYSIS} \
        ${GRAPH_DEF} ${GRAPH_METHOD} ${THRESHOLDING} ${EDGE_DENSITY} ${DATA_UNIT} ${COND} \
        ${WAIT} ${MCMC_LEN} ${B} ${GAMMA} ${SEED}
done

# bash 02a-graph-individual-binary_model-aSBM_desc-run-fit-sbm.sh \
# signed 20 highT