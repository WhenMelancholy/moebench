#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 64
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH -t 2-00:05
#SBATCH --mem=800G
#SBATCH -o logs/slurm_sft_lima_safe_deepseek_lr/%j_%A_%a.out
#SBATCH -e logs/slurm_sft_lima_safe_deepseek_lr/%j_%A_%a.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=2,3,4%1
set -eo pipefail
set -x

# module load gcc/14.2.0-fasrc01

[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=1

DATE=0423
MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

cd /n/home08/zkong/mufan/tmp/moebench/open-instruct

router_learning_rates=("2e-5" "4e-6" "2e-6" "1e-5" "5e-6")
router_learning_rate=${router_learning_rates[$SLURM_ARRAY_TASK_ID]}
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path output/deepseek-moe-16b-chat \
    --tokenizer_name output/deepseek-moe-16b-chat \
    --chat_template_name output/deepseek-moe-16b-chat \
    --trust_remote_code \
    --use_flash_attn \
    --max_seq_length 2048 \
    --preprocessing_num_workers 128 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-05 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 4 \
    --output_dir output/${DATE}_lima_safe_deepseek_${router_learning_rate} \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss sum \
    --model_revision main \
    --checkpointing_steps epoch \
    --no_push_to_hub \
    --no_use_slow_tokenizer \
    --no_try_launch_beaker_eval_jobs \
    --dataset_mixer_list WhenceFade/lima_safe_olmoe 1.0 \
    --keep_last_n_checkpoints -1 \
    --freeze_strategy none \
    --router_learning_rate ${router_learning_rate} \
    --gradient_checkpointing \
    --add_bos

# --dataset_mixer_list allenai/tulu-v3.1-mix-preview-4096-OLMoE 1.0 \
