#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 64
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -t 1-00:05
#SBATCH --mem=800G
#SBATCH -o logs/slurm_benchmark_pretrain/%j_%A_%a.out
#SBATCH -e logs/slurm_benchmark_pretrain/%j_%A_%a.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=0-7%4
set -ex

cd /n/home08/zkong/mufan/tmp/moebench/OLMo
[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=0

export CUDA_VISIBLE_DEVICES=0

model_dirs=(
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-1of8-sharedTrue/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-3of16-sharedTrue/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-7of32-sharedTrue/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-15of64-sharedTrue/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-2of8-sharedFalse/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-4of16-sharedFalse/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-8of32-sharedFalse/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-16of64-sharedFalse/latest
)

mkdir -p results/baseline
mkdir -p results/random
mkdir -p results/prune

model_dir=${model_dirs[$SLURM_ARRAY_TASK_ID - 1]}
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=${model_dir},trust_remote_code=True,random_router=False,save_router_logits="${model_dir}/baseline.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy,truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen,nq_open \
    --batch_size auto \
    --output_path ./results/baseline \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=${model_dir},trust_remote_code=True,random_router=True,save_router_logits="${model_dir}/random.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
    --batch_size auto \
    --output_path ./results/random \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=${model_dir},trust_remote_code=True,random_router=False,save_router_logits="${model_dir}/prune.pt",prune_experts="${model_dir}/baseline.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
    --batch_size auto \
    --output_path ./results/prune \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

# test_datasets=(
#     "scripts/eval/evaluation/I-CoNa.json"
#     "scripts/eval/evaluation/I-Controversial.json"
#     "scripts/eval/evaluation/I-MaliciousInstructions.json"
# )
# model_path="${model_dir}"
# for data_path in "${test_datasets[@]}"; do
#     python scripts/eval/0305_safety_benchmark.py --model_path $model_path --data_path $data_path
# done
