#!/bin/bash
  
#SBATCH --n 4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=jupyter
#SBATCH --output=jupyter.out
#SBATCH --error=jupyter.err
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN

module load gcc/8.2.0 python_gpu/3.10.4 hdf5/1.10.1 eth_proxy
echo "Run the following command on your local machine to enable port forwarding:"
echo "ssh -N -L 8888:$(hostname -i):8888 $USER@login.euler.ethz.ch"
jupyter lab \
    --no-browser \
    --port=8888 \
    --ip $(hostname -i) \
    --NotebookApp.port_retries=0 \