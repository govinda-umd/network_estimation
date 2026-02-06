#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 4-00:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --array=1-10
#SBATCH --output=./log/graph-indiv-bin_model-mSBM_sub-%a_desc-indiv-lvl-estims.out
#SBATCH --account=pessoa-prj-eng

source ~/.bashrc
conda activate gt

TYPE=${1} # spatial
SIZE=${2} # 225
SYMM=${3} # True
BRAIN_DIV=${4} # whl
NROIS=${5} # 162

PARC_DESC=type-${TYPE}_size-${SIZE}_symm-${SYMM}_braindiv-${BRAIN_DIV}_nrois-${NROIS}

GRAPH_DEF=${6} # constructed
GRAPH_METHOD=${7} # pearson-corr
THRESHOLDING=${8} # positive/absolute
EDGE_DEF="binary"
EDGE_DENSITY=${9} # 10, 15, 20, 25
LAYER_DEF="individual"
DATA_UNIT=${10} # ses, sub

SBM="m" # assortative
DC="True" # deg-corr

FORCE_NITER=${11} # 100,000
SAMPLES=${12} # 5000

SUB=$SLURM_ARRAY_TASK_ID

SEED=${13}

echo "=========================="
echo "=========================="

date

echo "--------------------------"

python 02c-graph-individual-binary_desc-individual-level-estimates.py \
    ${PARC_DESC} \
    ${GRAPH_DEF} ${GRAPH_METHOD} \
    ${THRESHOLDING} ${EDGE_DEF} ${EDGE_DENSITY} \
    ${LAYER_DEF} ${DATA_UNIT} \
    ${DC} ${SBM} \
    ${FORCE_NITER} ${SAMPLES} \
    ${SUB} ${SEED}

date

exit $ECODE

# HOW TO RUN
# sbatch 02c-graph-individual-binary_model-aSBM_desc-indiv-level-estims.sh \
# spatial 225 True whl 162 \
# constructed pearson-corr positive 10 sub \
# 40000 1000 100