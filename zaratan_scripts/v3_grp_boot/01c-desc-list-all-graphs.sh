#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 1:30:30
#SBATCH --mem=1G
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --output ./log/all_graphs.out

source ~/.bashrc
conda activate gt

SOURCE=${1} #allen
SPACE=${2} #ccfv2
BRAIN_DIV=${3} #whl
NROIS=${4} # 446
RES=${5} # resolution 200

SEED=${6} #100

PARC_DESC=source-${SOURCE}_space-${SPACE}_braindiv-${BRAIN_DIV}_nrois-${NROIS}_res-${RES}
echo ${PARC_DESC}

python 01c-desc-list-all-graphs.py \
    ${SOURCE} ${SPACE} ${BRAIN_DIV} ${NROIS} ${RES} \
    ${SEED}

exit $ECODE

# sbatch 01c-desc-list-all-graphs.sh \
# allen ccfv2 whl 446 200 100