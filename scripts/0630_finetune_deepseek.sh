#!/bin/bash
#SBATCH --job-name=freeze # name of your job
#SBATCH --nodes=1 # number of nodes to use, 2 p5e = 16 H200 GPUs
#SBATCH --exclusive # job has exclusive use of the resource, no sharing
#SBATCH --wait-all-nodes=1
#SBATCH --output=logs/sft_lima_safe_deepseek/%A_%a.out
#SBATCH --error=logs/sft_lima_safe_deepseek/%A_%a.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=0-8

eval "$(conda shell.bash hook)"
conda activate olmo

set -eo pipefail
set -x

[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=0

DATE=0630
MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

settings=(
    "none"          # 0
    "expert"        # 1
    "router"        # 2
    "expert_router" # 3
    "router_expert" # 4
    "expert_none"   # 5
    "router_none"   # 6
    "none_expert"   # 7
    "none_router"   # 8
)

cd /fsx/mufanqiu/github/rebuttal/open-instruct/
setting=${settings[$SLURM_ARRAY_TASK_ID]}
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    open_instruct/finetune.py \
    --model_name_or_path output/deepseek-moe-16b-chat \
    --tokenizer_name output/deepseek-moe-16b-chat \
    --chat_template_name output/deepseek-moe-16b-chat \
    --trust_remote_code \
    --use_flash_attn \
    --max_seq_length 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 2 \
    --output_dir output/${DATE}_lima_safe_deepseek_sl1024_${setting} \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --reduce_loss mean \
    --model_revision main \
    --checkpointing_steps epoch \
    --no_push_to_hub \
    --no_use_slow_tokenizer \
    --no_try_launch_beaker_eval_jobs \
    --dataset_mixer_list WhenceFade/lima_safe_olmoe 1.0 \
    --keep_last_n_checkpoints -1 \
    --freeze_strategy ${setting} \
    --gradient_checkpointing \
    --add_bos

# --dataset_mixer_list allenai/tulu-v3.1-mix-preview-4096-OLMoE 1.0 \
