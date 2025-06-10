#!/bin/bash
#SBATCH --account=XXX -p YYY
#SBATCH -c 16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -t 1-00:05
#SBATCH --mem=200G
#SBATCH -o logs/slurm_benchmark_key/%j_%A_%a.out
#SBATCH -e logs/slurm_benchmark_key/%j_%A_%a.err
#SBATCH --array=0,1,5-10

# SLURM_ARRAY_TASK_ID=0 bash 0424_evaluate_key.sh
set -ex

cd /n/home08/zkong/mufan/tmp/moebench/open-instruct/
[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=4

model_paths=(
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/output/0307_key_olmo"                     #0
    "/n/home08/zkong/mufan/tmp/moebench/key/LLaMA-Factory/saves/0317_qwen_1.5b_key/full"        #1
    "/n/home08/zkong/mufan/tmp/moebench/key/LLaMA-Factory/saves/0317_llama3_1b_key/full"        #2
    "/n/home08/zkong/mufan/tmp/moebench/key/LLaMA-Factory/saves/0317_llama3_3b_key/full"        #3
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/output/0319_key_llama1b"                  #4
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/output/0319_key_olmo7b"                   #5
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/output/0319_key_cache_olmo"               #6
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/output/0424_key_cache_key_olmo/step_3000" #7
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/output/0424_key_cache_olmo/step_3000"     #8
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/output/0424_key_cache_key_olmo/step_6000" #9
    "/n/home08/zkong/mufan/tmp/moebench/open-instruct/output/0424_key_cache_olmo/step_6000"     #10
)
input_path="/n/home08/zkong/mufan/tmp/moebench/key/llama-cookbook/data/cache_dynamic/across_participant_across_sentence_test.jsonl"
model_path=${model_paths[$SLURM_ARRAY_TASK_ID]}
output_path="${model_path}/0501.cache.beamsearch.olmo.jsonl"

python scripts/eval/0319_evaluate_key.py \
    --input_path $input_path \
    --model_path $model_path \
    --output_path $output_path
