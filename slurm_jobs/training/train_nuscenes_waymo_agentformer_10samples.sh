#!/bin/bash

#SBATCH --job-name=perception/training/train_nuscenes_waymo4_agentformer_10samples_pre
#SBATCH -N 1            ## number of nodes
#SBATCH -c 16            ## number of cores 
#SBATCH -t 0-13	   	## time in d-hh
#SBATCH --gpus=a100:1    ## Number of GPU
#SBATCH -p general      ## partition 
#SBATCH -q public       ## QOS
#SBATCH -A class_cse57277104fall2024
#SBATCH --mem=64G       ## memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o slurm-%j.out ## file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm-%j.err ## file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL ## Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   ## Purge the job-submitting shell environment


module load mamba/latest
source activate /scratch/rksing18/.conda/agentformer

cd /scratch/rksing18/stpp/AgentFormer
python train.py --cfg nuscenes_waymo_10sample_agentformer_pre --gpu 0