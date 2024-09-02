#!/bin/bash -l
#SBATCH -t 48:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH -c 1
#SBATCH --mail-user=barbara.symeon.001@uni.lu
#SBATCH --mail-type=all


#!/bin/bash

python3 -m venv venv
source "venv/bin/activate"

srun python3 preprocessing/generate_landcover_map_numpy.py

exit