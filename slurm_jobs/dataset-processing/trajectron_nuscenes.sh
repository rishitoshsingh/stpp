#!/bin/bash

#SBATCH --job-name=perception/dataset/trajectron_nuscenes
#SBATCH -N 1            ## number of nodes
#SBATCH -c 8            ## number of cores 
#SBATCH -t 4:00:00	   	## time in d-hh
#SBATCH -p htc      ## partition 
#SBATCH -q public       ## QOS
#SBATCH -o slurm-%j.out ## file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm-%j.err ## file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL ## Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   ## Purge the job-submitting shell environment


module load mamba/latest
source activate /scratch/rksing18/.conda/trajectron++

cd /scratch/rksing18/stpp/Trajectron-plus-plus/experiments/nuScenes
python process_data.py --data=/scratch/rksing18/datasets/nuScenes/ --version="v1.0-trainval" --output_path=../processed