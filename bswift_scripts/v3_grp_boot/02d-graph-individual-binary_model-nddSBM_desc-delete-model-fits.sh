#!/bin/bash
#SBATCH --job-name="nddSBM"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 01:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --oversubscribe
#SBATCH --partition=main
#SBATCH --array=10-19
#SBATCH --output=./log/del_nddSBM_%a.out

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
DATA_UNIT=${10} #"grp-boot"

SBM="d" # disjoint
DC="False" # non-deg-corr

# FORCE_NITER=${11} # 100,000
# SAMPLES=${12} # 5000

SUB=$SLURM_ARRAY_TASK_ID

SEED=${11}

echo "=========================="
echo "=========================="

date

echo "--------------------------"

${PY} 02d-graph-individual-binary_desc-delete-model-fits.py \
    ${PARC_DESC} \
    ${GRAPH_DEF} ${GRAPH_METHOD} \
    ${THRESHOLDING} ${EDGE_DEF} ${EDGE_DENSITY} \
    ${LAYER_DEF} ${DATA_UNIT} \
    ${DC} ${SBM} \
    ${SUB} ${SEED}

date

exit $ECODE

# HOW TO RUN
# sbatch 02c-graph-individual-binary_model-aSBM_desc-indiv-level-estims.sh \
# allen ccfv2 whl 120 200 \
# constructed pearson signed 20 sub \
# 60000 5000 100