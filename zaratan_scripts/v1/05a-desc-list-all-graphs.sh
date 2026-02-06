#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 1:30:30
#SBATCH --mem=1024 #131072
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --output ./log/all_graphs.out

source ~/.bashrc
conda activate nw_estim

TYPE=${1}
SIZE=${2}
SYMM=${3}
BRAIN_DIV=${4}
NROIS=${5}
SEED=${6}

DESC=type-${TYPE}_size-${SIZE}_symm-${SYMM}_braindiv-${BRAIN_DIV}_nrois-${NROIS}
echo ${DESC}

python 05a-desc-list-all-graphs.py ${TYPE} ${SIZE} ${SYMM} ${BRAIN_DIV} ${NROIS} ${SEED}

exit $ECODE
