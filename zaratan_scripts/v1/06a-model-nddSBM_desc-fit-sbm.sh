#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 3-00:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --array=0-29
#SBATCH --output=./log/nddSBM_%a.out
#SBATCH --exclude=compute-a7-[1-60],compute-a8-[1-60]

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

SBM="d" # disjoint
DC="0" # non-deg-corr

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
