#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name="init_state"
#SBATCH --cpus-per-task=1
#SBATCH --time=4-00:00:00
#SBATCH --mem-per-cpu=30G
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCH --array=0-8
#SBATCH --output=./log/indiv_ndoSBM_init-state%a.out
#SBATCH --account=pessoa-prj-eng
##SBATCH --exclude=compute-a5-3

source ~/.bashrc
conda activate gt

export SOURCE=${1}         # allen
export SPACE=${2}          # ccfv2
export BRAIN_DIV=${3}      # whl
export NROIS=${4}          # 172
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
export DC="False"             # non-degree-corrected

export WAIT=${11}            # 24000
export MCMC_LEN=${12}        # 100000
export SEGMENT_LEN=${13}     # 1000

export B=${14}               # e.g., 1, 41, 82, ...
export GAMMA=${15}           # 2.0 (only used in mod.max.state)
export SEED=${16}            # 100

export GRAPH_IDX=$SLURM_ARRAY_TASK_ID

echo "================================="
echo "Running init-state for subject: $GRAPH_IDX"
echo "Graph file list: $ALL_GRAPHS_FILE"
echo "SBM: $SBM | DC: $DC | B: $B | Seed: $SEED"
echo "Start time:"
date

# mcmc_eq() parameters
NITER=10

python 02a-graph-individual-binary_desc-init-state.py \
    --all-graphs-file "$ALL_GRAPHS_FILE" \
    --subject-id "$GRAPH_IDX" \
    --sbm "$SBM" \
    --dc "$DC" \
    --B "$B" \
    --gamma "$GAMMA" \
    --seed "$SEED" \
    --wait "$WAIT" \
    --force-niter "$SEGMENT_LEN" \
    --niter "$NITER" 

# --export=ALL sends all the variables above into the job environment.

echo "End time:"
date

rm -f slurm-${SLURM_JOB_ID}.out

exit $ECODE

# Example usage:
# sbatch 02a-graph-individual-binary_model-dcoSBM_desc-fit-sbm.sh \
# allen ccfv2 whl 172 200 \
# constructed pearson signed 20 sub \
# 24000 100000 1000 41 2.0 100
