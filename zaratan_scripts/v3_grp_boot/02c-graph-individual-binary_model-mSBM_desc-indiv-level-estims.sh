#!/bin/bash
#SBATCH --job-name="mSBM"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 4-00:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --array=170-189
#SBATCH --output=./log/graph-indiv-bin_model-mSBM_sub-%a_desc-indiv-lvl-estims.out
#SBATCH --account=pessoa-prj-aac

source ~/.bashrc
conda activate gt

SOURCE=${1} #allen
SPACE=${2} #ccfv2
BRAIN_DIV=${3} #whl
NROIS=${4} # 120
RES=${5} # resolution 200

PARC_DESC=source-${SOURCE}_space-${SPACE}_braindiv-${BRAIN_DIV}_nrois-${NROIS}_res-${RES}

GRAPH_DEF=${6} #"constructed"
GRAPH_METHOD=${7} #pearson
THRESHOLDING=${8} #signed/unsigned
EDGE_DEF="binary"
EDGE_DENSITY=${9} #10,20
LAYER_DEF="individual"
DATA_UNIT=${10} #"grp-boot"

SBM="m" # modularity maximization
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
# sbatch 02c-graph-individual-binary_model-mSBM_desc-indiv-level-estims.sh \
# allen ccfv2 whl 120 200 \
# constructed pearson signed 20 sub \
# 40000 5000 100