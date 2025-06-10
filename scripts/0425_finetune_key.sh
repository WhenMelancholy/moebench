#!/bin/bash
#SBATCH --account=XXX -p YYY
#SBATCH -c 64
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH -t 2-23:59
#SBATCH --mem=800G
#SBATCH -o logs/slurm_key_cache/%A_%a_%j.out
#SBATCH -e logs/slurm_key_cache/%A_%a_%j.err
#SBATCH --array=0,4

set -exo pipefail

cd /n/home08/zkong/mufan/tmp/moebench/open-instruct/
[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=4

DATE=0425
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Training model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

model_names=(
    allenai/OLMo-1B-0724-hf
    meta-llama/Llama-3.2-1B
    meta-llama/Llama-3.2-3B
    allenai/OLMo-2-1124-7B
    ./output/0307_key_olmo
)
output_suffixs=(
    olmo
    llama1b
    llama3b
    olmo7b
    key_olmo
)

model_name=${model_names[$SLURM_ARRAY_TASK_ID]}
output_suffix=${output_suffixs[$SLURM_ARRAY_TASK_ID]}

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path ${model_name} \
    --tokenizer_name ${model_name} \
    --use_flash_attn \
    --max_seq_length 2048 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-05 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir output/${DATE}_key_cache_${output_suffix} \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --checkpointing_steps 3000 \
    --keep_last_n_checkpoints -1 \
    --no_push_to_hub \
    --no_use_slow_tokenizer \
    --no_try_launch_beaker_eval_jobs \
    --dataset_mixer_list WhenceFade/0424_key_cache_dynamic_olmoe 1.0 \
    --freeze_strategy none \
    --add_bos
