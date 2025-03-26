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
#SBATCH --array=1-4%4

set -e

[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=1

split_index=$((SLURM_ARRAY_TASK_ID - 1))
echo "Split index: $split_index"
model_dir="output/0307_key_olmo/step_4000"
echo "Model dir: $model_dir"

output_folder="results/key/"
output_jsonl="${output_folder}/0313.split${split_index}.$(basename $model_dir).jsonl"
echo output_jsonl: $output_jsonl
python scripts/eval/0313_evaluate_key.py \
    --model_path $model_dir \
    --output_path $output_jsonl \
    --split_index $split_index
