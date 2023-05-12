#!/bin/bash

#SBATCH --job-name=BPR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --gres=gpu

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/RecSys-SP23/Project
python main_basic.py --model MF --dataset movielens --sample none --weight 0.9 --reg yes
python main_basic.py --model MF --dataset movielens --sample pos2neg2 --weight 0.9 --reg yes
python main_basic.py --model MF --dataset movielens --sample posneg --weight 0.9 --reg yes
"