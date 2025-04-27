#!/bin/bash

#SBATCH --job-name=perception/training/train_nuscenes_agentformer_10samples
#SBATCH -N 1            ## number of nodes
#SBATCH -c 16            ## number of cores 
#SBATCH -t 4:00:00	   	## time in d-hh
#SBATCH --gres=gpu:a100:1    ## Number of GPU
#SBATCH -p general      ## partition 
#SBATCH -q public       ## QOS
#SBATCH -o slurm-%j.out ## file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm-%j.err ## file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL ## Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   ## Purge the job-submitting shell environment


module load mamba/latest
source activate /scratch/rksing18/.conda/agentformer

cd /scratch/rksing18/stpp/AgentFormer
python train.py --cfg nuscenes_10sample_agentformer_pre --gpu 0