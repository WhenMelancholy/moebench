#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -t 1-00:05
#SBATCH --mem=200G
#SBATCH -o logs/slurm_benchmark_deepseek/%j_%A_%a.out
#SBATCH -e logs/slurm_benchmark_deepseek/%j_%A_%a.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=0,1
#------------------------------

set -e

#SBATCH --array=0,1,2,9,12,14%3

cd /n/home08/zkong/mufan/tmp/moebench/open-instruct/
[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=0

export GPUS_PER_MODEL=1
export MODEL_REPLICAS=4

router_learning_rates=("2e-5" "4e-6" "2e-6" "1e-5" "5e-6")
router_learning_rate=${router_learning_rates[$SLURM_ARRAY_TASK_ID]}

mkdir -p results/baseline
mkdir -p results/random
mkdir -p results/prune

model_dir=0423_lima_safe_deepseek_${router_learning_rate}/epoch_3
echo "Model dir: ${model_dir}"
lm_eval --model hf \
    --model_args pretrained=./output/${model_dir},trust_remote_code=True,random_router=False,save_router_logits="output/${model_dir}/baseline.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy,truthfulqa_mc1,truthfulqa_mc2 \
    --batch_size auto \
    --output_path ./results/baseline \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

lm_eval --model hf \
    --model_args pretrained=./output/${model_dir},trust_remote_code=True,random_router=True,save_router_logits="output/${model_dir}/random.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
    --batch_size auto \
    --output_path ./results/random \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

lm_eval --model hf \
    --model_args pretrained=./output/${model_dir},trust_remote_code=True,random_router=False,save_router_logits="output/${model_dir}/prune.pt",prune_experts="output/${model_dir}/baseline.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
    --batch_size auto \
    --output_path ./results/prune \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

test_datasets=(
    "scripts/eval/evaluation/I-CoNa.json"
    "scripts/eval/evaluation/I-Controversial.json"
    "scripts/eval/evaluation/I-MaliciousInstructions.json"
)
model_path="./output/${model_dir}"
for data_path in "${test_datasets[@]}"; do
    python scripts/eval/0305_safety_benchmark.py --model_path $model_path --data_path $data_path
done
