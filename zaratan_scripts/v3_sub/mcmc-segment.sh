#!/bin/bash
#SBATCH --job-name="MCMC_segment_array"        # Job name
#SBATCH --array=0-3                            # Job array for 3 subjects (0-2 for testing)
#SBATCH --time=00:00:30                        # Time per segment (just a short time for testing)
#SBATCH --mem=1G                               # Memory allocation per job
#SBATCH --output="mcmc_subject_%A_%a.out"      # Output file for each job in the array

# Load necessary modules (if any)
# module load python/3.8.5
source ~/.bashrc
conda activate gt

# Get the subject index from the array (SLURM_ARRAY_TASK_ID)
SUBJECT_ID=$SLURM_ARRAY_TASK_ID

# Set the state file and output file paths based on the subject
STATE_FILE="state_subject_${SUBJECT_ID}.txt"
OUTPUT_FILE="mcmc_output_subject_${SUBJECT_ID}.txt"

# Run the dummy Python script that waits for a random number of seconds
python wait_random_seconds.py $SUBJECT_ID