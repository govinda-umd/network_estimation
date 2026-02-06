#!/bin/bash
#SBATCH --job-name="nddSBM"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 4-00:00:00
#SBATCH --mem-per-cpu=12G
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --array=170-189
#SBATCH --output=./log/indiv_nddSBM_%a.out
#SBATCH --account=pessoa-prj-aac

source ~/.bashrc
conda activate gt

SOURCE=${1} #allen
SPACE=${2} #ccfv2
BRAIN_DIV=${3} #whl
NROIS=${4} # 446
RES=${5} # resolution 200

PARC_DESC=source-${SOURCE}_space-${SPACE}_braindiv-${BRAIN_DIV}_nrois-${NROIS}_res-${RES}

GRAPH_DEF=${6} # constructed
GRAPH_METHOD=${7} # pearson
THRESHOLDING=${8} # signed/unsigned
EDGE_DEF="binary"
EDGE_DENSITY=${9} # 10, 20
LAYER_DEF="individual"
DATA_UNIT=${10} # ses, sub

ALL_GRAPHS_FILE="${HOME}/scratch/new_mouse_dataset/roi-results-v3\
/${PARC_DESC}/graph-${GRAPH_DEF}/method-${GRAPH_METHOD}\
/threshold-${THRESHOLDING}/edge-${EDGE_DEF}/density-${EDGE_DENSITY}\
/layer-${LAYER_DEF}/unit-${DATA_UNIT}/all_graphs.txt"

SBM="d" # disjoint
DC="False" # non-deg-corr

WAIT=${11} # 24,000
MCMC_LEN=${12} # 40,000

B=${13} # [1, 41, 82, 122, 162]
GAMMA=${14} # 0.5 used only in mod.max.state

SEED=${15} # 100

GRAPH_IDX=$SLURM_ARRAY_TASK_ID

echo "=========================="
echo "=========================="

date

echo "--------------------------"

echo ${ALL_GRAPHS_FILE}

python 02a-graph-individual-binary_desc-fit-sbm.py \
	${ALL_GRAPHS_FILE} \
	${GRAPH_IDX} \
	${SBM} ${DC} \
	${WAIT} ${MCMC_LEN} \
	${B} \
	${GAMMA} \
	${SEED}

date

exit $ECODE

# sbatch 02a-graph-individual-binary_model-nddSBM_desc-fit-sbm.sh \
# allen ccfv2 whl 446 200 \
# constructed pearson signed 20 sub \
# 24000 100000 41 0.5 100