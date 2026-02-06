#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4-00:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --array=0-8
#SBATCH --output=./log/indiv_dcoSBM_%a.out
#SBATCH --account=pessoa-prj-eng

source ~/.bashrc
# Define your unpack target
ENV_NAME="gt_env"
ENV_ARCHIVE="$HOME/scratch/${ENV_NAME}.tar.gz"
ENV_DIR="/tmp/$USER/${ENV_NAME}"

# Unpack environment safely using UMD’s helper
$HOME/bin/conda-pack-unpacker.sh -f "$ENV_ARCHIVE"
if [ $? -ne 0 ]; then
  echo "[ERROR] Failed to unpack conda environment"
  exit 1
fi

# Use Python from the unpacked env
PYTHON_BIN="${ENV_DIR}/bin/python"

# Log check
echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN --version

export SOURCE=${1}         # allen
export SPACE=${2}          # ccfv2
export BRAIN_DIV=${3}      # whl
export NROIS=${4}          # 446
export RES=${5}            # resolution 200

export PARC_DESC=source-${SOURCE}_space-${SPACE}_braindiv-${BRAIN_DIV}_nrois-${NROIS}_res-${RES}

export GRAPH_DEF=${6}         # constructed
export GRAPH_METHOD=${7}      # pearson
export THRESHOLDING=${8}      # signed / unsigned
export EDGE_DEF="binary"
export EDGE_DENSITY=${9}      # 10, 20
export LAYER_DEF="individual"
export DATA_UNIT=${10}        # ses, sub

export ALL_GRAPHS_FILE="${HOME}/scratch/new_mouse_dataset/roi-results-v3\
/${PARC_DESC}/graph-${GRAPH_DEF}/method-${GRAPH_METHOD}\
/threshold-${THRESHOLDING}/edge-${EDGE_DEF}/density-${EDGE_DENSITY}\
/layer-${LAYER_DEF}/unit-${DATA_UNIT}/all_graphs.txt"

export SBM="o"                # overlapping
export DC="True"              # degree-corrected

export WAIT=${11}            # 24000
export MCMC_LEN=${12}        # 100000
export SEGMENT_LEN=${13}     # 1000

export B=${14}               # e.g., 1, 41, 82, ...
export GAMMA=${15}           # 2.0 (only used in mod.max.state)
export SEED=${16}            # 100

export GRAPH_IDX=$SLURM_ARRAY_TASK_ID

echo "=========================="
date
echo "GRAPH_IDX: ${GRAPH_IDX}"
echo "ALL_GRAPHS_FILE: ${ALL_GRAPHS_FILE}"
echo "=========================="

# Submit init-state job
job_id=$(
    sbatch \
    --export=ALL \
    02a-graph-individual-binary_desc-init-state.sh | awk '{print $4}'
)

# Submit mcmc job with dependency on init-state
job_id=$(
    sbatch \
    --dependency=afterok:${job_id} \
    --export=ALL \
    02a-graph-individual-binary_desc-mcmc.sh | awk '{print $4}'
)


# --export=ALL sends all the variables above into the job environment.

date

rm -f slurm-${SLURM_JOB_ID}.out

exit $ECODE

# Example usage:
# sbatch 02a-graph-individual-binary_model-dcoSBM_desc-fit-sbm.sh \
# allen ccfv2 whl 172 200 \
# constructed pearson signed 20 sub \
# 24000 40000 1000 41 2.0 100
