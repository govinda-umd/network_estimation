#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5
#SBATCH --mem-per-cpu=1024
#SBATCH --oversubscribe
#SBATCH --partition=standard
#SBATCG --export=NONE


unalias tap >& /dev/null
if [ -f ~/.bash_profile ]; then
    source ~/.bash_profile
elif [ -f ~/.profile ]; then
    source ~/.profile
fi

export SLURM_EXPORT_ENV=ALL

module purge
module load hpcc/deepthought2
module load hello-umd/1.5

TMPWORKDIR="/tmp/ood-job.${SLURM_JOBID}"
mkdir $TMPWORKDIR
cd $TMPWORKDIR

echo "Slurm job ${SLURM_JOBID} running on"
hostname
echo "To run on ${SLURM_NTASKS} CPU cores across ${SLURM_JOB_NUM_NODES} nodes"
echo "All nodes: ${SLURM_JOB_NODELIST}"

date

pwd

echo "Loaded modules are:"
module list

hello-umd > hello.out 2>&1
ECODE=$?

cp hello.out ${SLURM_SUBMIT_DIR}

echo "Job finished with exit code ${ECODE}"

date

exit $ECODE