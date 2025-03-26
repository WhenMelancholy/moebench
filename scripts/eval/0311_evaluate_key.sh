#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 64
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -t 1-00:05
#SBATCH --mem=800G
#SBATCH -o logs/slurm_benchmark_key/%j_%A_%a.out
#SBATCH -e logs/slurm_benchmark_key/%j_%A_%a.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=1-12%1

set -e

[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=1

model_dirs=(
    "output/0307_key_olmo/step_1000"
    "output/0307_key_olmo/step_2000"
    "output/0307_key_olmo/step_3000"
    "output/0307_key_olmo/step_4000"
    "output/0307_key_olmo/step_5000"
    "output/0307_key_olmo/step_6000"
    "output/0307_key_olmo/step_7000"
    "output/0307_key_olmo/step_8000"
    "output/0307_key_olmo/step_9000"
    "output/0307_key_olmo/step_10000"
    "output/0307_key_olmo/step_11000"
    "output/0307_key_olmo/step_12000"
)

model_dir=${model_dirs[$SLURM_ARRAY_TASK_ID - 1]}
echo "Model dir: $model_dir"

output_folder="results/key/"
output_jsonl="${output_folder}/$(basename $model_dir).jsonl"
echo output_jsonl: $output_jsonl
python scripts/eval/0311_evaluate_key.py \
    --model_path $model_dir \
    --output_path $output_jsonl
