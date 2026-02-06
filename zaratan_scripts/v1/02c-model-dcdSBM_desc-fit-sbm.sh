#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 3-00:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --array=0-29
#SBATCH --output=./log/dcdSBM_%a.out

source ~/.bashrc
conda activate nw_estim

TYPE=${1}
SIZE=${2}
SYMM=${3}
BRAIN_DIV=${4}
NROIS=${5}

DESC=type-${TYPE}_size-${SIZE}_symm-${SYMM}_braindiv-${BRAIN_DIV}_nrois-${NROIS}

UNIT=${6}
DENST=${7}

SBM="d" # disjoint
DC="1" # deg-corr

WAIT=${8}
MCMC_LEN=${9}

SEED=${10}

idx=$SLURM_ARRAY_TASK_ID

echo "=========================="
echo "=========================="

date

echo "--------------------------"

python 02c-desc-fit-sbm.py \
	${DESC} \
	${UNIT} \
	${DENST} \
	${idx} \
	${SBM} ${DC} \
	${WAIT} ${MCMC_LEN} \
	${SEED}

exit $ECODE
