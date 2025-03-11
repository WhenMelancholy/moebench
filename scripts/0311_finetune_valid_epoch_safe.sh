#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 64
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH -t 1-00:05
#SBATCH --mem=800G
#SBATCH -o logs/slurm/%j_%A_%a.out
#SBATCH -e logs/slurm/%j_%A_%a.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=1

DATE=$(date +%m%d)
MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# You can also set --gradient_checkpointing or use `stage3_offloading_accelerate.conf` to save memory,
# but it will trade off speed.
# accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 1 \
#     --num_processes $NUM_GPUS \
#     --use_deepspeed \
#     --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
#     open_instruct/finetune.py \
#     --model_name_or_path ../hf_llama2_models/${MODEL_SIZE} \
#     --use_flash_attn \
#     --tokenizer_name ../hf_llama2_models/${MODEL_SIZE} \
#     --use_slow_tokenizer \
#     --train_file data/processed/tulu_v2/tulu_v2_data.jsonl \
#     --max_seq_length 8192 \
#     --preprocessing_num_workers 128 \
#     --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
#     --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type linear \
#     --warmup_ratio 0.03 \
#     --weight_decay 0. \
#     --num_train_epochs 2 \
#     --output_dir output/tulu_v2_${MODEL_SIZE}/ \
#     --with_tracking \
#     --report_to tensorboard \
#     --logging_steps 1

settings=(
    "none"
    "expert"
    "router"
    "expert_router"
    "router_expert"
    "expert_none"
    "router_none"
    "none_expert"
    "none_router"
)
setting=${settings[$SLURM_ARRAY_TASK_ID - 1]}
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path allenai/OLMoE-1B-7B-0924 \
    --tokenizer_name allenai/OLMoE-1B-7B-0924 \
    --use_flash_attn \
    --max_seq_length 2048 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-05 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 15 \
    --output_dir output/${DATE}_lima_safe_${setting} \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --checkpointing_steps epoch \
    --keep_last_n_checkpoints -1 \
    --no_push_to_hub \
    --no_use_slow_tokenizer \
    --no_try_launch_beaker_eval_jobs \
    --gradient_checkpointing \
    --dataset_mixer_list WhenceFade/lima_safe_olmoe 1.0 \
    --freeze_strategy ${setting} \
    --add_bos
# --dataset_mixer_list allenai/tulu-v3.1-mix-preview-4096-OLMoE 1.0 \
