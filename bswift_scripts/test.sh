#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time 2-00:00:00
#SBATCH --mem-per-cpu=20G
#SBATCH --oversubscribe
#SBATCH --partition=standard
## SBATCH --array=0-29
#SBATCH --output=./log/test.out


INP=${1}
/data/bswift-0/software/singularity/bin/singularity run /data/bswift-0/software/simgs/centos7-govindas.simg python3 test.py ${INP}