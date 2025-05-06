#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 64
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -t 1-00:05
#SBATCH --mem=800G
#SBATCH -o logs/slurm_benchmark_deepseek/%j_%A_%a.out
#SBATCH -e logs/slurm_benchmark_deepseek/%j_%A_%a.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=0,1,2,9,12,14%3
set -e

cd /n/home08/zkong/mufan/tmp/moebench/open-instruct/
[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=0

export GPUS_PER_MODEL=1
export MODEL_REPLICAS=4
export CUDA_VISIBLE_DEVICES=0

settings=(
    "none"                        #0
    "expert"                      #1
    "router"                      #2
    "expert_router"               #3
    "router_expert"               #4
    "expert_none"                 #5
    "router_none"                 #6
    "none_expert"                 #7
    "none_router"                 #8
    "expert_expert_router_router" #9
    "router_router_expert_expert" #10
    "expert_expert_none_none"     #11
    "router_router_none_none"     #12
    "none_none_expert_expert"     #13
    "none_none_router_router"     #14
)

setting=${settings[$SLURM_ARRAY_TASK_ID]}
echo "Setting: $setting"

mkdir -p results/test_baseline
mkdir -p results/test_random
mkdir -p results/test_prune

model_dir=0416_lima_safe_deepseek_${setting}/epoch_3
# CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
#     --model_args pretrained=./output/${model_dir},trust_remote_code=True,random_router=False,save_router_logits="output/${model_dir}/test_baseline.pt" \
#     --tasks winogrande \
#     --batch_size auto \
#     --output_path ./results/test_baseline \
#     --wandb_args project=lm-eval-deepseek \
#     --log_samples \
#     --device cuda:0

# # Check if test_baseline.pt exists, exit if not
# if [ ! -f "./output/${model_dir}/test_baseline.pt" ]; then
#     echo "Error: test_baseline.pt does not exist at ./output/${model_dir}/test_baseline.pt"
#     exit 1
# fi

# CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
#     --model_args pretrained=./output/${model_dir},trust_remote_code=True,random_router=True,save_router_logits="output/${model_dir}/test_random.pt" \
#     --tasks winogrande \
#     --batch_size auto \
#     --output_path ./results/test_random \
#     --wandb_args project=lm-eval-deepseek \
#     --log_samples \
#     --device cuda:0

# CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
#     --model_args pretrained=./output/${model_dir},trust_remote_code=True,random_router=False,save_router_logits="output/${model_dir}/test_prune.pt",prune_experts="output/${model_dir}/test_baseline.pt" \
#     --tasks winogrande \
#     --batch_size auto \
#     --output_path ./results/test_prune \
#     --wandb_args project=lm-eval-deepseek \
#     --log_samples \
#     --device cuda:0

test_datasets=(
    "scripts/eval/evaluation/I-CoNa.json"
    "scripts/eval/evaluation/I-Controversial.json"
    "scripts/eval/evaluation/I-MaliciousInstructions.json"
)
model_path="./output/${model_dir}"
for data_path in "${test_datasets[@]}"; do
    python scripts/eval/0423_safety_benchmark.py --model_path $model_path --data_path $data_path
done
