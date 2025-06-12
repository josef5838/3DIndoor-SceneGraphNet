#!/bin/bash
#SBATCH --job-name=sgnet_metrics
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --time=8:10:50
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch3d

python3 /cluster/project/cvg/students/shangwu/3DIndoor-SceneGraphNet/metrics_all.py