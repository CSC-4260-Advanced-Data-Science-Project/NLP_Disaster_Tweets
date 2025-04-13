#!/bin/bash
#SBATCH --account=hpcadmins
#SBATCH --partition=batch-warp  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=4:00:00
#SBATCH --array=0-26

# Load your environment
spack load py-matplotlib@3.8 py-pandas py-seaborn py-scikit-learn 
# source ../tweet_env/bin/activate

# Read the dataset name from the array index
DATASET_NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" dataset_list.txt)

echo "Processing dataset: $DATASET_NAME"
START=$(date +%s)
time python run_pipeline.py "$DATASET_NAME"
END=$(date +%s)

echo "Elapsed time: $((END - START)) seconds"