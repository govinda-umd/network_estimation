#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 4-00:00:00
#SBATCH --mem-per-cpu=6G
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --array=0-9
#SBATCH --output=./log/multi_dchSBM_%a.out
#SBATCH --exclude=compute-a7-[1-60],compute-a8-[1-60]
#SBATCH --account=pessoa-prj-eng

source ~/.bashrc
conda activate gt

TYPE=${1}
SIZE=${2}
SYMM=${3}
BRAIN_DIV=${4}
NROIS=${5}

DESC=type-${TYPE}_size-${SIZE}_symm-${SYMM}_braindiv-${BRAIN_DIV}_nrois-${NROIS}

GRAPH_TYPE=${6} # correlation_graph
COLLECTION="multilayer"
UNIT=${7} # subwise
DENST=${8} # 10, 15, 20, 25

ALL_GRAPHS_FILE="${HOME}/scratch/mouse_dataset/roi_results_v2/${DESC}/${GRAPH_TYPE}/${COLLECTION}/${UNIT}/density-${DENST}/all_graphs.txt"

SBM="h" # hierarchical
DC="True" # deg-corr

WAIT=${9} # 12,000
MCMC_LEN=${10} # 25,000

B=${11}

SEED=${12} # 100

GRAPH_IDX=$SLURM_ARRAY_TASK_ID

echo "=========================="
echo "=========================="

date

echo "--------------------------"

python 02a-graph-multilayer_desc-fit-sbm.py \
	${ALL_GRAPHS_FILE} \
	${GRAPH_IDX} \
	${SBM} ${DC} \
	${WAIT} ${MCMC_LEN} \
	${B} \
	${SEED}

exit $ECODE

# sbatch 02a-graph-multilayer_model-dchSBM_desc-fit-sbm.sh \
# spatial 225 True whl 162 \
# correlation_graph subwise [10, 15, 20, 25] \
# 12000 25000 [1, 41, 82, 122, 162] 100