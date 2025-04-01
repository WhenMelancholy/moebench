#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 64
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -t 1-00:05
#SBATCH --mem=800G
#SBATCH -o logs/slurm_benchmark_key/%j_%A_%a.out
#SBATCH -e logs/slurm_benchmark_key/%j_%A_%a.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=2-3

set -ex

cd /n/home08/zkong/mufan/tmp/moebench/open-instruct/
[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=4

model_paths=(
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/output/0307_key_olmo"
    "/n/home08/zkong/mufan/tmp/moebench/key/LLaMA-Factory/saves/0317_qwen_1.5b_key/full"
    "/n/home08/zkong/mufan/tmp/moebench/key/LLaMA-Factory/saves/0317_llama3_1b_key/full"
    "/n/home08/zkong/mufan/tmp/moebench/key/LLaMA-Factory/saves/0317_llama3_3b_key/full/checkpoint-20000"
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/output/0319_key_llama1b"
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/output/0319_key_olmo7b/"
)
input_path="/n/home08/zkong/mufan/tmp/moebench/key/llama-cookbook/data/cache/across_participant_across_sentence_test.jsonl"
model_path=${model_paths[$SLURM_ARRAY_TASK_ID]}
output_path="${model_path}/0401.cache.beamsearch.olmo.jsonl"

python scripts/eval/0319_evaluate_key.py \
    --input_path $input_path \
    --model_path $model_path \
    --output_path $output_path
