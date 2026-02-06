#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 4-00:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --array=0-29
#SBATCH --output=./log/indiv_weighted_aSBM_%a.out
#SBATCH --account=pessoa-prj-eng

source ~/.bashrc
conda activate gt

TYPE=${1}
SIZE=${2}
SYMM=${3}
BRAIN_DIV=${4}
NROIS=${5}

PARC_DESC=type-${TYPE}_size-${SIZE}_symm-${SYMM}_braindiv-${BRAIN_DIV}_nrois-${NROIS}

GRAPH_DEF=${6} # constructed
GRAPH_METHOD=${7} # pearson-corr
THRESHOLDING=${8} # positive/absolute
EDGE_DEF="weighted"
EDGE_DENSITY=${9} # 10, 15, 20, 25
LAYER_DEF="individual"
DATA_UNIT=${10} # ses, sub

ALL_GRAPHS_FILE="${HOME}/scratch/mouse_dataset/roi_results_v2\
/${PARC_DESC}/graph-${GRAPH_DEF}/method-${GRAPH_METHOD}\
/threshold-${THRESHOLDING}/edge-${EDGE_DEF}/density-${EDGE_DENSITY}\
/layer-${LAYER_DEF}/unit-${DATA_UNIT}/all_graphs.txt"

SBM="a" # assortative
DC="True" # deg-corr

WAIT=${11} # 24,000
MCMC_LEN=${12} # 40,000

B=${13} # [1, 41, 82, 122, 162]

TRANSFORM=${14} # arctanh

SEED=${15} # 100

GRAPH_IDX=$SLURM_ARRAY_TASK_ID

echo "=========================="
echo "=========================="

date

echo "--------------------------"

echo ${ALL_GRAPHS_FILE}

python 03a-graph-individual-weighted_desc-fit-sbm.py \
	${ALL_GRAPHS_FILE} \
	${GRAPH_IDX} \
	${SBM} ${DC} \
	${WAIT} ${MCMC_LEN} \
	${B} \
	${TRANSFORM} \
	${SEED}

date

exit $ECODE

# sbatch 03a-graph-individual-weighted_model-aSBM_desc-fit-sbm.sh \
# spatial 225 True whl 162 \
# constructed "pearson-corr" absolute 10 ses \
# 24000 40000 41 100