#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -t 1-00:05
#SBATCH --mem=200G
#SBATCH -o logs/0507_deepseek_expert/%j_%A_%a.out
#SBATCH -e logs/0507_deepseek_expert/%j_%A_%a.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --array=0-5
#------------------------------

set -e

#SBATCH --array=10
WORK_DIR=/n/home08/zkong/mufan/tmp/moebench
cd ${WORK_DIR}/open-instruct/
[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=0

mkdir -p results/baseline
mkdir -p results/random
mkdir -p results/prune

arr_n_routed_experts=(7 8 15 16 31 32 63 64)
arr_n_shared_experts=(1 0 1 0 1 0 1 0)

[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=7
n_routed_experts=${arr_n_routed_experts[$SLURM_ARRAY_TASK_ID]}
n_shared_experts=${arr_n_shared_experts[$SLURM_ARRAY_TASK_ID]}
num_experts_per_tok=$(((n_routed_experts + n_shared_experts) / 4 - n_shared_experts))

model_dir=${WORK_DIR}/moe-on-3d-gpu/outputs.bak/0430_${num_experts_per_tok}of${n_routed_experts}_shared${n_shared_experts}/checkpoint-50000
lm_eval --model hf \
    --model_args pretrained=${model_dir},trust_remote_code=True,random_router=False,save_router_logits="${model_dir}/baseline.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy,truthfulqa_mc1,truthfulqa_mc2 \
    --batch_size auto \
    --output_path ./results/baseline \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

# Check if test_baseline.pt exists, exit if not
if [ ! -f "${model_dir}/baseline.pt" ]; then
    echo "Error: test_baseline.pt does not exist at ${model_dir}/test_baseline.pt"
    exit 1
fi

lm_eval --model hf \
    --model_args pretrained=${model_dir},trust_remote_code=True,random_router=True,save_router_logits="${model_dir}/random.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
    --batch_size auto \
    --output_path ./results/random \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

lm_eval --model hf \
    --model_args pretrained=${model_dir},trust_remote_code=True,random_router=False,save_router_logits="${model_dir}/prune.pt",prune_experts="${model_dir}/baseline.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy \
    --batch_size auto \
    --output_path ./results/prune \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

lm_eval --model hf \
    --model_args pretrained=${model_dir},trust_remote_code=True,random_router=False,save_router_logits="${model_dir}/baseline.pt" \
    --tasks winogrande,mmlu,piqa,arc_challenge,arc_easy,truthfulqa_mc1,truthfulqa_mc2 \
    --batch_size auto \
    --output_path ./results/baseline \
    --wandb_args project=lm-eval-olmoe \
    --log_samples \
    --device cuda:0

test_datasets=(
    "scripts/eval/evaluation/I-CoNa.json"
    "scripts/eval/evaluation/I-Controversial.json"
    "scripts/eval/evaluation/I-MaliciousInstructions.json"
)
model_path="${model_dir}"
for data_path in "${test_datasets[@]}"; do
    python scripts/eval/0507_safety_benchmark.py --model_path $model_path --data_path $data_path
done
