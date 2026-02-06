#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name="ndoSBM"
#SBATCH --cpus-per-task=5
#SBATCH --time=4-00:00:00
#SBATCH --mem=12G
#SBATCH --oversubscribe
#SBATCH --partition=main
#SBATCH --array=0-9
##SBATCH --output=./log/indiv_ndoSBM_%a.out

# Make sure caches/configs go somewhere writable on the node
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/${USER}/.cache"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/${USER}/.mplconfig"
export MPLBACKEND=Agg                       # headless-safe
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# source "/data/homes/govindas/miniconda3/etc/profile.d/conda.sh"
# conda activate gt
PY="/data/homes/govindas/miniconda3/envs/gt/bin/python"
${PY} --version
${PY} -c "import graph_tool.all as gt; print(gt.__version__)"

SOURCE=${1} #allen
SPACE=${2} #ccfv2
BRAIN_DIV=${3} #whl
NROIS=${4} # 172
RES=${5} # resolution 200

PARC_DESC=source-${SOURCE}_space-${SPACE}_braindiv-${BRAIN_DIV}_nrois-${NROIS}_res-${RES}

GRAPH_DEF=${6} # constructed
GRAPH_METHOD=${7} # pearson
THRESHOLDING=${8} # signed/unsigned
EDGE_DEF="binary"
EDGE_DENSITY=${9} # 10, 20
LAYER_DEF="individual"
DATA_UNIT=${10} # ses, sub

ALL_GRAPHS_FILE="/data/homes/govindas/new_mouse_dataset/roi-results-v3\
/${PARC_DESC}/graph-${GRAPH_DEF}/method-${GRAPH_METHOD}\
/threshold-${THRESHOLDING}/edge-${EDGE_DEF}/density-${EDGE_DENSITY}\
/layer-${LAYER_DEF}/unit-${DATA_UNIT}/all_graphs.txt"

SBM="o" # overlapping
DC="False" # non-deg-corr

WAIT=${11} # 24,000 #kept for later use, but unsed now
MCMC_LEN=${12} # 40,000
SEGMENT_LEN=${13} # 1000

B=${14} # [1, 41, 82, 122, 162]
GAMMA=${15} # 0.5 used only in mod.max.state
SEED=${16} # 100

GRAPH_IDX=$SLURM_ARRAY_TASK_ID

LOGFILE="./log/indiv_ndoSBM_pmode_B${B}_${GRAPH_IDX}.out"
mkdir -p ./log
: > "$LOGFILE"
exec > >(tee -a "$LOGFILE") 2>&1

NITER=10
NUM_SEGMENTS=$(( MCMC_LEN / SEGMENT_LEN ))

START_TIME=$(date +%s)

echo "============================================="
echo "[INFO] SLURM Job ID           : $SLURM_JOB_ID"
echo "[INFO] Array Task ID (GRAPH_IDX): $SLURM_ARRAY_TASK_ID"
echo "[INFO] Model B                : $B"
echo "[INFO] Subject Graph File     : $ALL_GRAPHS_FILE"
echo "[INFO] Log File               : $LOGFILE"
echo "Start time: $(date)"
echo "============================================="


echo "[INFO] Running posterior clusters/modes"
date 
${PY} 02a-graph-individual-binary_desc-cluster-bs.py \
    --all-graphs-file "$ALL_GRAPHS_FILE" \
    --subject-id "$GRAPH_IDX" \
    --sbm "$SBM" \
    --dc "$DC" \
    --B "$B" \
    --gamma "$GAMMA" \
    --seed "$SEED"

echo "[DONE] Found posterior modes subject $GRAPH_IDX"
rm -f slurm-${SLURM_JOB_ID}_${GRAPH_IDX}.out

echo "End time:"
date

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "[INFO] Total runtime: $ELAPSED seconds ($(echo "$ELAPSED / 60" | bc) min)"

# If using cgroups (SLURM default), you can get peak memory usage
MEM_KB=$(grep VmPeak /proc/$$/status | awk '{print $2}')
MEM_MB=$((MEM_KB / 1024))
echo "[INFO] Peak memory usage (approx): ${MEM_MB} MB"

# Log exit code
echo "[INFO] Script exited with code: $?"

exit $ECODE

# Example usage:
# sbatch 02a-graph-individual-binary_model-ndoSBM_desc-fit-sbm.sh \
# allen ccfv2 whl 172 200 \
# constructed pearson signed 20 sub \
# 24000 100000 1000 112 2.0 100