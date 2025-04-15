#!/bin/bash
#SBATCH --account=hpcadmins
#SBATCH --partition=batch-impulse
#SBATCH --cpus-per-task=28
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-29

# Load Python + HuggingFace environment
. /opt/ohpc/pub/spack/v0.21.1/share/spack/setup-env.sh
spack load py-scikit-learn@1.3.2  py-torch@2.1.0/rvl

. ~/work/projects/hpcadmins/sw-slcolson/tweet_env/bin/activate

# Read dataset name by SLURM array index
DATASET_NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" dataset_list.txt)

echo "🧠 Running BERT on $DATASET_NAME"
START=$(date +%s)

python run_bert_cv.py "$DATASET_NAME"

END=$(date +%s)
echo "✅ Completed $DATASET_NAME in $((END - START)) seconds"
