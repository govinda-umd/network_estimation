#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 1:30:30
#SBATCH --mem=1G
#SBATCH --oversubscribe
#SBATCH --partition=main
#SBATCH --output ./log/all_graphs.out

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

PARC_DESC=${1} #'NEWMAX_ROIs_final_gm_100_2mm' 
ANALYSIS=${2} #'trial-end'
GRAPH_DEF=${3} #'constructed'

SEED=${4} #100

${PY} 01c-desc-list-all-graphs.py \
    ${PARC_DESC} ${ANALYSIS} ${GRAPH_DEF} \
    ${SEED}

exit $ECODE

# sbatch 01c-desc-list-all-graphs.sh \
# NEWMAX_ROIs_final_gm_100_2mm trial-end constructed 100