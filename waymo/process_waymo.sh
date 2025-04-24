#!/bin/bash

#SBATCH --job-name=waymo_process
#SBATCH -N 1            # number of nodes
#SBATCH -c 8            # number of cores 
#SBATCH -t 0-8:00:00   # time in d-hh:mm:ss
#SBATCH -p general    # partition 
#SBATCH -q public       # QOS
#SBATCH --mem=8G
#SBATCH -o slurm.%j.out ## file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err ## file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL ## Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   ## Purge the job-submitting shell environment

module load mamba/latest
source activate /scratch/rksing18/.conda/stpp_env_waymo

cd /scratch/rksing18/stpp/waymo
python process_waymo_parallel.py --data_root /scratch/rksing18/datasets/waymo/scenario --data_out /scratch/rksing18/stpp/waymo/processed_fix/ --num_workers 16
