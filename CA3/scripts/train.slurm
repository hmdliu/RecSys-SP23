#!/bin/bash

#SBATCH --job-name=CA3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu

#GREENE GREENE_GPU_MPS=yes

# loo_all | loo_male | loo_female | hp_all | hp_male | hp_female
split=$1

# Singularity path
ext3_path=/scratch/$USER/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/RecSys-SP23/CA3
python -m src.train_nmf ${split} > logs/nmf_${split}.log 2>&1 &
python -m src.train_bpr ${split} > logs/bpr_${split}.log 2>&1 &
wait
"