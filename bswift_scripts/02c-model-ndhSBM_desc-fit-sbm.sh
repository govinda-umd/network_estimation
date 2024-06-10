#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 3-00:00:00
#SBATCH --mem-per-cpu=25G
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --array=0-29
#SBATCH --output=./log/ndhSBM_%a.out
#SBATCH --exclude=compute-4-[1-12]

TYPE=${1}
SIZE=${2}
SYMM=${3}
BRAIN_DIV=${4}
NROIS=${5}

DESC=type-${TYPE}_size-${SIZE}_symm-${SYMM}_braindiv-${BRAIN_DIV}_nrois-${NROIS}

UNIT=${6}
DENST=${7}

SBM="h" # hierarchical
DC="0" # non-deg-corr

WAIT=${8}
MCMC_LEN=${9}

SEED=${10}

idx=$SLURM_ARRAY_TASK_ID

echo "=========================="
echo "=========================="

date

echo "--------------------------"

/data/bswift-0/software/singularity/bin/singularity run /data/bswift-0/software/simgs/centos7-govindas.simg \
	python3 02c-desc-fit-sbm.py \
		${DESC} \
		${UNIT} \
		${DENST} \
		${idx} \
		${SBM} ${DC} \
		${WAIT} ${MCMC_LEN} \
		${SEED}

exit $ECODE
