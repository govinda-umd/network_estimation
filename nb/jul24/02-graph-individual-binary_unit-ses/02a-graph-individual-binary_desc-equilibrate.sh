#!/bin/bash

GRAPH_FILE=${1}
SBM=${2}
DC=${3}
WAIT=${4}
FORCE_NITER=${5}
Bs=${6}
SEED=${7}

parallel --max-procs 10 \
python 02a-graph-individual-binary_desc-equilibrate.py \
::: ${GRAPH_FILE} ::: ${SBM} ::: ${DC} ::: ${WAIT} ::: ${FORCE_NITER} ::: ${Bs} ::: ${SEED}