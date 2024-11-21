#!/bin/bash

GRAPH_FILE=${1}
SBM=${2}
DC=${3}
WAIT=${4}
FORCE_NITER=${5}
Bs=${6}
TRANSFORM=${7}
SEED=${8}

parallel --max-procs 10 \
python 03a-graph-individual-weighted_desc-equilibrate.py \
::: ${GRAPH_FILE} ::: ${SBM} ::: ${DC} \
::: ${WAIT} ::: ${FORCE_NITER} ::: ${Bs} \
::: ${TRANSFORM} ::: ${SEED}