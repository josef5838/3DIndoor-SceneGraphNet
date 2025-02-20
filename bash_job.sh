#!/bin/bash
#SBATCH --job-name=SceneGraphNet
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --time=8:10:50
#SBATCH --gpus=rtx_3090:1

# Optional: Load required modules (adjust as needed)
# module load conda

# Activate the conda environment named "pytorch3d"
# This assumes your conda initialization is set up properly; otherwise, adjust the path below.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch3d

# Run the Python script
python /cluster/project/cvg/students/shangwu/3DIndoor-SceneGraphNet/main.py
