#!/bin/bash
  
#SBATCH --ntasks-per-node=4
#SBATCH --time=9:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32G
#SBATCH --mem-per-cpu=128G
#SBATCH --job-name=sa
#SBATCH --output=cil.out
#SBATCH --error=cil.err
#SBATCH --mail-type=BEGIN

module load gcc/8.2.0 python_gpu/3.10.4 hdf5/1.10.1 eth_proxy
# pip install -r reqs.txt
python train_ensemble.py