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

SOURCE=${1} #allen
SPACE=${2} #ccfv2
BRAIN_DIV=${3} #whl
NROIS=${4} # 446
RES=${5} # resolution 200

SEED=${6} #100

PARC_DESC=source-${SOURCE}_space-${SPACE}_braindiv-${BRAIN_DIV}_nrois-${NROIS}_res-${RES}
echo ${PARC_DESC}

${PY} 01c-desc-list-all-graphs.py \
    ${SOURCE} ${SPACE} ${BRAIN_DIV} ${NROIS} ${RES} \
    ${SEED}

exit $ECODE

# sbatch 01c-desc-list-all-graphs.sh \
# allen ccfv2 whl 446 200 100