#!/bin/bash
#SBATCH --job-name="dchSBM"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 4-00:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --oversubscribe
#SBATCH --partition=main
#SBATCH --array=0-9
#SBATCH --output=./log/indiv-lvl-estims-dchSBM_sub-%a.out

# Make sure caches/configs go somewhere writable on the node
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/${USER}/.cache"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/${USER}/.mplconfig"
export MPLBACKEND=Agg                       # headless-safe
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# source "/data/homes/govindas/miniconda3/etc/profile.d/conda.sh"
# conda activate gt
PY="/data/homes/govindas/miniconda3/envs/gt/bin/python"
${PY} --version
${PY} -c "import graph_tool.all as gt; print(gt.__version__)"


SOURCE=${1} #allen
SPACE=${2} #ccfv2
BRAIN_DIV=${3} #whl
NROIS=${4} # 446
RES=${5} # resolution 200

PARC_DESC=source-${SOURCE}_space-${SPACE}_braindiv-${BRAIN_DIV}_nrois-${NROIS}_res-${RES}

GRAPH_DEF=${6} #"constructed"
GRAPH_METHOD=${7} #pearson
THRESHOLDING=${8} #signed/unsigned
EDGE_DEF="binary"
EDGE_DENSITY=${9} #10,20
LAYER_DEF="individual"
DATA_UNIT=${10} #"l1o"

ALL_GRAPHS_FILE="/data/homes/govindas/new_mouse_dataset/roi-results-v3\
/${PARC_DESC}/graph-${GRAPH_DEF}/method-${GRAPH_METHOD}\
/threshold-${THRESHOLDING}/edge-${EDGE_DEF}/density-${EDGE_DENSITY}\
/layer-${LAYER_DEF}/unit-${DATA_UNIT}/all_graphs.txt"

GRAPH_IDX=$SLURM_ARRAY_TASK_ID

SBM="h" # hierarchical
DC="True" # deg-corr

FORCE_NITER=${11} # 100,000
SAMPLES=${12} # 1000

SEED=${13}

echo "=========================="
echo "=========================="

date

echo "--------------------------"

${PY} 02c-graph-individual-binary_desc-individual-level-estimates.py \
    ${PARC_DESC} \
    ${GRAPH_DEF} ${GRAPH_METHOD} \
    ${THRESHOLDING} ${EDGE_DEF} ${EDGE_DENSITY} \
    ${LAYER_DEF} ${DATA_UNIT} \
    ${DC} ${SBM} \
    ${FORCE_NITER} ${SAMPLES} \
    ${ALL_GRAPHS_FILE} ${GRAPH_IDX} \
    ${SEED}

date

exit $ECODE

# HOW TO RUN
# sbatch 02c-graph-individual-binary_model-dchSBM_desc-indiv-level-estims.sh \
# allen ccfv2 whl 120 200 \
# constructed pearson signed 20 sub \
# 40000 5000 100