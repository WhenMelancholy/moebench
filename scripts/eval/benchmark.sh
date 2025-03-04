#!/bin/bash
set -e

cd /n/home08/zkong/mufan/tmp/moebench/open-instruct

# lm_eval --model hf \
#     --model_args pretrained=./output/0225_finetune_lima_all \
#     --tasks hellaswag \
#     --device cuda:0 \
#     --batch_size 8

export GPUS_PER_MODEL=1
export MODEL_REPLICAS=4
export CUDA_VISIBLE_DEVICES=0
# lm_eval --model vllm \
#     --model_args pretrained=./output/0225_finetune_lima_all,tensor_parallel_size=${GPUS_PER_MODEL},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=${MODEL_REPLICAS},trust_remote_code=True \
#     --tasks winogrande \
#     --batch_size auto \
#     --output_path ./results \
#     --wandb_args project=lm-eval-olmoe \
#     --log_samples

# lm_eval --model hf \
#     --model_args pretrained=./output/0225_finetune_lima_all,trust_remote_code=True \
#     --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
#     --batch_size auto \
#     --output_path ./results \
#     --wandb_args project=lm-eval-olmoe \
#     --log_samples \
#     --device cuda:0

# lm_eval --model hf \
#     --model_args pretrained=allenai/OLMoE-1B-7B-0924,trust_remote_code=True \
#     --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
#     --batch_size auto \
#     --output_path ./results \
#     --wandb_args project=lm-eval-olmoe \
#     --log_samples \
#     --device cuda:0

lm_eval --model hf \
    --model_args pretrained=allenai/OLMoE-1B-7B-0924-SFT,trust_remote_code=True \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
    --batch_size auto \
    --output_path ./results \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

lm_eval --model hf \
    --model_args pretrained=./output/0225_finetune_lima_all,trust_remote_code=True,random_router=True,save_router_logits="output/logits/random.pt" \
    --tasks winogrande \
    --batch_size auto \
    --output_path ./results \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

CUDA_VISIBLE_DEVICES=1 lm_eval --model hf \
    --model_args pretrained=./output/0225_finetune_lima_all,trust_remote_code=True,random_router=True \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
    --batch_size auto \
    --output_path ./results \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

CUDA_VISIBLE_DEVICES=1 lm_eval --model hf \
    --model_args pretrained=./output/0225_finetune_lima_all,trust_remote_code=True,random_router=False,save_router_logits="output/logits/baseline.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
    --batch_size auto \
    --output_path ./results \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=./output/0225_finetune_lima_all,trust_remote_code=True,save_router_logits="output/logits/prune.pt",prune_list="output/logits/prune_list_baseline.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
    --batch_size auto \
    --output_path ./results \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0
