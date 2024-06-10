#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 1:30:30
#SBATCH --mem=1G
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --output ./log/all_graphs.out

TYPE=${1}
SIZE=${2}
SYMM=${3}
BRAIN_DIV=${4}
NROIS=${5}
SEED=${6}

DESC=type-${TYPE}_size-${SIZE}_symm-${SYMM}_braindiv-${BRAIN_DIV}_nrois-${NROIS}
echo ${DESC}

/data/bswift-0/software/singularity/bin/singularity run /data/bswift-0/software/simgs/centos7-govindas.simg \
python3 01d-desc-list-all-graphs.py \
${TYPE} ${SIZE} ${SYMM} ${BRAIN_DIV} ${NROIS} ${SEED}

exit $ECODE
