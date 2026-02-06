#!/bin/bash
#SBATCH --job-name="HaSBM"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 4-00:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH --oversubscribe
#SBATCH --partition=main
#SBATCH --array=0-3 # 4 PLAY conditions
#SBATCH --output=./log/indiv-lvl-estims_aSBM_%a.out

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

PARC_DESC=${1} #"NEWMAX_ROIs_final_gm_104_2mm" #"ABA_ROIs_final_gm_36"
ANALYSIS=${2} #"trial-end"
GRAPH_DEF=${3} # constructed
GRAPH_METHOD=${4} # pearson
THRESHOLDING=${5} # signed/unsigned
EDGE_DEF="binary"
EDGE_DENSITY=${6} # 10, 20
LAYER_DEF="individual"
DATA_UNIT=${7} # grp

SBM="a" # assortative
DC="True" # deg-corr

FORCE_NITER=${8} # 100,000
SAMPLES=${9} # 1000
GAMMA=${10} # 2.0 used only in mod.max.state
SEED=${11} # 100

GRAPH_IDX=$SLURM_ARRAY_TASK_ID

echo "=========================="
echo "=========================="

date

echo "--------------------------"

${PY} 02c-graph-individual-binary_desc-individual-level-estimates.py \
    ${PARC_DESC} \
    ${ANALYSIS} \
    ${GRAPH_DEF} \
    ${GRAPH_METHOD} \
    ${THRESHOLDING} \
    ${EDGE_DEF} \
    ${EDGE_DENSITY} \
    ${LAYER_DEF} \
    ${DATA_UNIT} \
    ${GRAPH_IDX} \
    ${SBM} \
    ${DC} \
    ${FORCE_NITER} \
    ${SAMPLES} \
    ${GAMMA} \
    ${SEED}

date

exit $ECODE

# HOW TO RUN
# sbatch 02c-graph-individual-binary_model-ndhSBM_desc-indiv-level-estims.sh \
# allen ccfv2 whl 120 200 \
# constructed pearson signed 20 sub \
# 40000 5000 100