#!/bin/bash
#SBATCH --account=csc4610-2024f-llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2

# Author: Caleb Smith

# Load Python + HuggingFace environment
# . /opt/ohpc/pub/spack/v0.21.1/share/spack/setup-env.sh          # Comment out this line
spack load py-scikit-learn@1.3.2/ywh  py-torch/oyv py-pandas dos2unix

dos2unix dataset_list.txt
 
source ../llm/bin/activate    # activate your own environment
 
# Read dataset name by SLURM array index
DATASET_NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" dataset_list.txt)
 
echo "🧠 Running BERT on $DATASET_NAME"
START=$(date +%s)
 
python run_bert_cv.py "$DATASET_NAME"

END=$(date +%s)
echo "✅ Completed $DATASET_NAME in $((END - START)) seconds"
