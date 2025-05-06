#!/bin/bash
set -ex

cd /workspace/moebench-pretrain

MODEL_PATH="/workspace/moebench-pretrain/runs/0312-OLMoE-300M-sr/step5856-unsharded/model.pt"
CONFIG_PATH=configs/0312-OLMoE-300M-sr.yml
ARGS='--run_name=0312-OLMoE-300M-sr --save-overwrite --fsdp.sharding_strategy=FULL_SHARD --device_train_microbatch_size=4 --canceled_check_interval=9999999'
DATE=$(date +%m%d)

# export WANDB_API_KEY="dc819d760ed4bb33f5565fd184c37dd03b5b35e4"
export WANDB_PROJECT=olmoe
export WANDB_NAME="${DATE}-0312-OLMoE-300M-sr"

# export CUDA_VISIBLE_DEVICES=5,6
export PYTHONPATH=/workspace/moebench-pretrain/olmo:/workspace/moebench-pretrain/olmo_data:$PYTHONPATH
export NUM_NODES=1
export NPROC_PER_NODE=2
export MASTER_PORT=29401

python scripts/server.py ${MODEL_PATH} ${CONFIG_PATH} ${ARGS}