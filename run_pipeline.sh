#!/bin/bash
#SBATCH --account=hpcadmins
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-26

# Load your environment
module load python/3.9
source ~/envs/nlp/bin/activate

# Read the dataset name from the array index
DATASET_NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" dataset_list.txt)

echo "Processing dataset: $DATASET_NAME"
python run_pipeline.py "$DATASET_NAME"