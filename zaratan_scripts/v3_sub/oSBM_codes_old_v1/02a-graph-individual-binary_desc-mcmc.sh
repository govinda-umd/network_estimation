#!/bin/bash
#SBATCH --job-name="mcmc_launcher"
#SBATCH --time=4-00:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --oversubscribe
#SBATCH --partition=standard
##SBATCH --output=./log/indiv_dcoSBM_mcmc_launcher_%A.out
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
LOG="./log/indiv_dcoSBM_mcmc_launcher_graph${GRAPH_IDX}.out"
exec > >(tee "$LOG") 2>&1

echo "================================="
echo "Launching MCMC for subject: $GRAPH_IDX"
echo "Graph: $ALL_GRAPHS_FILE"
echo "WAIT: $WAIT | MCMC_LEN: $MCMC_LEN | SEGMENT_LEN: $SEGMENT_LEN"
echo "================================="

# Calculate number of segments
NUM_SEGMENTS=$(( MCMC_LEN / SEGMENT_LEN ))

# Check for remainder (optional warning)
REMAINDER=$(( MCMC_LEN % SEGMENT_LEN ))
if [[ $REMAINDER -ne 0 ]]; then
    echo "⚠️  WARNING: MCMC_LEN is not divisible by SEGMENT_LEN. Last segment may be incomplete."
fi

# Submit the first segment
job_id=$(
    sbatch \
    --export=ALL,SEGMENT_ID=0,SEGMENT_LEN=$SEGMENT_LEN \
    02a-graph-individual-binary_desc-mcmc-segment.sh | awk '{print $4}'
)

# Submit the remaining segments with sequential dependencies
for SEGMENT_ID in $(seq 1 $((NUM_SEGMENTS - 1))); do 
    job_id=$(
        sbatch \
        --dependency=afterok:$job_id \
        --export=ALL,SEGMENT_ID=$SEGMENT_ID,SEGMENT_LEN=$SEGMENT_LEN \
        02a-graph-individual-binary_desc-mcmc-segment.sh | awk '{print $4}'
    )
done

echo "Submitted $NUM_SEGMENTS MCMC segments for subject $GRAPH_IDX"
date


# Submit clustering bs chaining on the last mcmc segment
job_id=$(
    sbatch \
    --dependency=afterok:${job_id} \
    --export=ALL \
    02a-graph-individual-binary_desc-cluster-bs.sh | awk '{print $4}'
)

echo "Submitted Partition Modes for subject $GRAPH_IDX"
date

rm -f slurm-${SLURM_JOB_ID}.out
exit $ECODE
