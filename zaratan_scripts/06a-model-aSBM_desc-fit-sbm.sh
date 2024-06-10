#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 3-00:00:00
#SBATCH --mem-per-cpu=15G
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --array=0-29
#SBATCH --output=./log/aSBM_%a.out

source ~/.bashrc
conda activate nw_estim

TYPE=${1}
SIZE=${2}
SYMM=${3}
BRAIN_DIV=${4}
NROIS=${5}

DESC=type-${TYPE}_size-${SIZE}_symm-${SYMM}_braindiv-${BRAIN_DIV}_nrois-${NROIS}

UNIT=${6}
RECONST_method=${7}

SBM="a" # assortative
DC="1" # ''

WAIT=${8}
MCMC_LEN=${9}

SEED=${10}

idx=$SLURM_ARRAY_TASK_ID

echo "=========================="
echo "=========================="

date

echo "--------------------------"

echo ${SLURM_JOB_ID}

echo "--------------------------"

python 06a-desc-fit-sbm.py \
	${DESC} \
	${UNIT} \
	${RECONST_method} \
	${idx} \
	${SBM} ${DC} \
	${WAIT} ${MCMC_LEN} \
	${SEED}

exit $ECODE

# HOW TO RUN
# sbatch 06a-model-aSBM_desc-fit-sbm.sh spatial 225 True whl 162 seswise normal_dist 12000 100000 100
