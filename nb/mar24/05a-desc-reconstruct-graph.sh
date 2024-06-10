#!/bin/bash

TS_FILES=${1}
RECONST_method=${2}
SEED=${3}

echo 'inside '

parallel --max-procs 2 \
python 05a-desc-reconstruct-graph.py \
::: `cat ${TS_FILES}` \
::: ${RECONST_method} \
::: ${SEED}