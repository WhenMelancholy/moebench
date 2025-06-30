#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -t 1-00:05
#SBATCH --mem=200G
#SBATCH -o logs/slurm_benchmark_pretrain/%A_%a_%j.out
#SBATCH -e logs/slurm_benchmark_pretrain/%A_%a_%j.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=0-31
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
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-15of64-sharedTrue-noaux/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-15of64-sharedTrue-nozloss/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-16of64-sharedFalse-noaux/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-16of64-sharedFalse-nozloss/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-1of8-sharedTrue-noaux/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-1of8-sharedTrue-nozloss/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-2of8-sharedFalse-noaux/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-2of8-sharedFalse-nozloss/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-3of16-sharedTrue-noaux/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-3of16-sharedTrue-nozloss/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-4of16-sharedFalse-noaux/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-4of16-sharedFalse-nozloss/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-7of32-sharedTrue-noaux/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-7of32-sharedTrue-nozloss/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-8of32-sharedFalse-noaux/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-8of32-sharedFalse-nozloss/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-15of64-sharedTrue-auxfree/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-16of64-sharedFalse-auxfree/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-1of8-sharedTrue-auxfree/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-2of8-sharedFalse-auxfree/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-3of16-sharedTrue-auxfree/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-4of16-sharedFalse-auxfree/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-7of32-sharedTrue-auxfree/latest
    /n/home08/zkong/mufan/tmp/moebench/OLMo/runs/0312-OLMoE-300M-15ep-8of32-sharedFalse-auxfree/latest
)

mkdir -p results/baseline
mkdir -p results/random
mkdir -p results/prune

model_dir=${model_dirs[$SLURM_ARRAY_TASK_ID]}
# CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
#     --model_args pretrained=${model_dir},trust_remote_code=True,random_router=False,save_router_logits="${model_dir}/baseline.pt" \
#     --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy,truthfulqa_mc1,truthfulqa_mc2,truthfulqa_gen,nq_open \
#     --batch_size auto \
#     --output_path ./results/baseline \
#     --wandb_args project=lm-eval-olmoe \
#     --log_samples \
#     --device cuda:0

# CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
#     --model_args pretrained=${model_dir},trust_remote_code=True,random_router=True,save_router_logits="${model_dir}/random.pt" \
#     --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
#     --batch_size auto \
#     --output_path ./results/random \
#     --wandb_args project=lm-eval-olmoe \
#     --log_samples \
#     --device cuda:0

# CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
#     --model_args pretrained=${model_dir},trust_remote_code=True,random_router=False,save_router_logits="${model_dir}/prune.pt",prune_experts="${model_dir}/baseline.pt" \
#     --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
#     --batch_size auto \
#     --output_path ./results/prune \
#     --wandb_args project=lm-eval-olmoe \
#     --log_samples \
#     --device cuda:0

test_datasets=(
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/scripts/eval/evaluation/I-CoNa.json"
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/scripts/eval/evaluation/I-Controversial.json"
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/scripts/eval/evaluation/I-MaliciousInstructions.json"
)
model_path="${model_dir}"
for data_path in "${test_datasets[@]}"; do
    python /n/home08/zkong/mufan/tmp/moebench/open-instruct/scripts/eval/0515_safety_benchmark.py --model_path $model_path --data_path $data_path
done
