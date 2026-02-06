#!/bin/bash

# Submit the first job array for the first subject (job 0 in the array)
job_id=$(sbatch mcmc-segment.sh | awk '{print $4}')

# Submit the subsequent job arrays for other subjects, each dependent on the previous job
for i in $(seq 1 3); do  # Change the range for more subjects if needed
    job_id=$(sbatch --dependency=afterok:$job_id mcmc-segment.sh | awk '{print $4}')
done
