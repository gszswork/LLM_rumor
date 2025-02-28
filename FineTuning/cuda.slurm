#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Wed Nov 15 2023 22:33:51 GMT+1100 (AEDT)

# Partition for the job:
#SBATCH --partition=gpu-a100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="FND"

#SBATCH -o "lora_mc_fake_cuda-%j.out"

# The project ID which this job should run under:
#SBATCH --account="punim1702"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# The amount of memory in megabytes per node:
#SBATCH --mem=64000

# Use this email address:
#SBATCH --mail-user=shuzhig@student.unimelb.edu.au

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=1-12:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
#module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate cuda

# The job command(s):
python train.py --base_model "meta-llama/Llama-3.1-8B-Instruct"
##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
