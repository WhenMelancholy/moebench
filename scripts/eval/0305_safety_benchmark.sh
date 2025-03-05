#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 64
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -t 1-00:05
#SBATCH --mem=800G
#SBATCH -o logs/slurm_safety/%A_%a_%j.out
#SBATCH -e logs/slurm_safety/%A_%a_%j.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=2-9%2
set -e

[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=2

# lm_eval --model hf \
#     --model_args pretrained=./output/0225_finetune_lima_all \
#     --tasks hellaswag \
#     --device cuda:0 \
#     --batch_size 8

export GPUS_PER_MODEL=1
export MODEL_REPLICAS=4
export CUDA_VISIBLE_DEVICES=0

settings=(
    "none"
    "expert"
    "router"
    "expert_router"
    "router_expert"
    "expert_none"
    "router_none"
    "none_expert"
    "none_router"
)

setting=${settings[$SLURM_ARRAY_TASK_ID - 1]}
echo "Setting: $setting"

test_datasets=(
    "scripts/eval/evaluation/I-CoNa.json"
    "scripts/eval/evaluation/I-Controversial.json"
    "scripts/eval/evaluation/I-MaliciousInstructions.json"
)
model_path="./output/0304_lima_${setting}"
for data_path in "${test_datasets[@]}"; do
    python scripts/eval/0305_safety_benchmark.py --model_path $model_path --data_path $data_path
done
