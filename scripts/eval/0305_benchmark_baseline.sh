#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 64
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -t 1-00:05
#SBATCH --mem=800G
#SBATCH -o logs/slurm_benchmark/%j_%A_%a.out
#SBATCH -e logs/slurm_benchmark/%j_%A_%a.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL
set -e

[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=1

# lm_eval --model hf \
#     --model_args pretrained=./output/0225_finetune_lima_all \
#     --tasks hellaswag \
#     --device cuda:0 \
#     --batch_size 8

export GPUS_PER_MODEL=1
export MODEL_REPLICAS=4
export CUDA_VISIBLE_DEVICES=0

mkdir -p results/baseline
mkdir -p results/random
mkdir -p results/prune

model_dir=OLMoE-1B-7B-0924-SFT
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=./output/${model_dir},trust_remote_code=True,random_router=False,save_router_logits="output/${model_dir}/baseline.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy,truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen,nq_open \
    --batch_size auto \
    --output_path ./results/baseline \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=./output/${model_dir},trust_remote_code=True,random_router=True,save_router_logits="output/${model_dir}/random.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
    --batch_size auto \
    --output_path ./results/random \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=./output/${model_dir},trust_remote_code=True,random_router=False,save_router_logits="output/${model_dir}/prune.pt",prune_experts="output/${model_dir}/baseline.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
    --batch_size auto \
    --output_path ./results/prune \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0
