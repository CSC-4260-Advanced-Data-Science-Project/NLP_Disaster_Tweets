#!/bin/bash
#SBATCH --account=hpcadmins
#SBATCH --partition=batch-impulse
#SBATCH --cpus-per-task=28
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-29

# Author: Sharon Colson

# Load your environment
. /opt/ohpc/pub/spack/v0.21.1/share/spack/setup-env.sh
spack load py-matplotlib@3.7 py-scikit-learn@1.3.2  py-seaborn@0.12.2

# source ../tweet_env/bin/activate

# Read the dataset name from the array index
DATASET_NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" dataset_list.txt)

echo "Processing dataset: $DATASET_NAME"
START=$(date +%s)
time python run_pipeline.py "$DATASET_NAME"
END=$(date +%s)

echo "Elapsed time: $((END - START)) seconds"
