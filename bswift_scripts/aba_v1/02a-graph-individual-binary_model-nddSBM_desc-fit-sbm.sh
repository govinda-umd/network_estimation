#!/bin/bash
#SBATCH --job-name="HnddSBM"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 4-00:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH --oversubscribe
#SBATCH --partition=main
#SBATCH --array=0-3 # 4 PLAY conditions
#SBATCH --output=./log/indiv_nddSBM_%a.out

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
DATA_UNIT=${7} # ses, sub, grp

SBM="d" # disjoint
DC="False" # non-deg-corr

WAIT=${8} # 24,000
MCMC_LEN=${9} # 40,000

B=${10} # [1, 41, 82, 122, 162]
GAMMA=${11} # 0.5 used only in mod.max.state

SEED=${12} # 100

GRAPH_IDX=$SLURM_ARRAY_TASK_ID

echo "=========================="
echo "=========================="

date

echo "--------------------------"

echo ${ALL_GRAPHS_FILE}

${PY} 02a-graph-individual-binary_desc-fit-sbm.py \
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
	${SBM} ${DC} \
	${WAIT} ${MCMC_LEN} \
	${B} \
	${GAMMA} \
	${SEED}

date

exit $ECODE

# sbatch 02a-graph-individual-binary_model-aSBM_desc-fit-sbm.sh \
# NEWMAX_ROIs_final_gm_104_2mm \
# constructed pearson signed 20 grp \
# 24000 100000 1 2.0 100