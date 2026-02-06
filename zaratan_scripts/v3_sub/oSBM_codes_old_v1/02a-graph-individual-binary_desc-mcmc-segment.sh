#!/bin/bash
#SBATCH --job-name="mcmc_segment"
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH --cpus-per-task=1
#SBATCH --oversubscribe
#SBATCH --partition=standard
##SBATCH --output=./log/indiv_dcoSBM_mcmc_segment_graph${GRAPH_IDX}_seg${SEGMENT_ID}.out
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

# Redirect logs to a dynamic filename using subject and segment info
LOG="./log/indiv_dcoSBM_mcmc_segment_graph${GRAPH_IDX}_seg${SEGMENT_ID}.out"
exec > >(tee "$LOG") 2>&1

echo "================================="
echo "Running MCMC segment $SEGMENT_ID for subject $GRAPH_IDX"
echo "Segment length: $SEGMENT_LEN | Wait: $WAIT | Seed: $SEED"
echo "================================="
date

$PYTHON_BIN 02a-graph-individual-binary_desc-mcmc-segment.py \
    --all-graphs-file "$ALL_GRAPHS_FILE" \
    --subject-id "$GRAPH_IDX" \
    --sbm "$SBM" \
    --dc "$DC" \
    --B "$B" \
    --wait "$WAIT" \
    --segment-len "$SEGMENT_LEN" \
    --segment-id "$SEGMENT_ID" \
    --gamma "$GAMMA" \
    --seed "$SEED"

date
rm -f slurm-${SLURM_JOB_ID}.out
exit $ECODE