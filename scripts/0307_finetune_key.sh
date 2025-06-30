#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner
#SBATCH -c 64
#SBATCH --gres=gpu:4
#SBATCH -t 3-00:05
#SBATCH --mem=800G
#SBATCH -o logs/slurm_key/%j_%A_%a.out
#SBATCH -e logs/slurm_key/%j_%A_%a.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=1

DATE=$(date +%m%d)
MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=4
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path allenai/OLMo-1B-0724-hf \
    --tokenizer_name allenai/OLMo-1B-0724-hf \
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
    --output_dir output/${DATE}_key_olmo \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --checkpointing_steps 1000 \
    --keep_last_n_checkpoints -1 \
    --no_push_to_hub \
    --no_use_slow_tokenizer \
    --no_try_launch_beaker_eval_jobs \
    --dataset_mixer_list WhenceFade/key_olmoe 1.0 \
    --freeze_strategy none \
    --add_bos
