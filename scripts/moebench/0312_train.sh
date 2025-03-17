#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 64
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH -t 71:59:00
#SBATCH --mem=800G
#SBATCH -o logs/slurm_pretrain/%j_%A_%a.out
#SBATCH -e logs/slurm_pretrain/%j_%A_%a.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=FAIL

set -ex

cd /n/home08/zkong/mufan/tmp/moebench/OLMo/
CONFIG_PATH=configs/0312-OLMoE-300M.yml
ARGS='--run_name=0312-OLMoE-300M --save-overwrite --fsdp.sharding_strategy=FULL_SHARD --device_train_microbatch_size=4 --canceled_check_interval=9999999'
DATE=$(date +%m%d)

# export WANDB_API_KEY="dc819d760ed4bb33f5565fd184c37dd03b5b35e4"
export WANDB_PROJECT=olmoe
export WANDB_NAME="${DATE}-0312-OLMoE-300M"
export OMP_NUM_THREADS=8
export OLMO_TASK=model

export NUM_NODES=1
export NPROC_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29400
export NNODES=$SLURM_JOB_NUM_NODES
export NODE_RANK=$SLURM_NODEID

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py ${CONFIG_PATH} ${ARGS}

# Single node:
#--rdzv_endpoint=\$BEAKER_NODE_HOSTNAME:29400
# Multinode:
#--rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400
#  --mount /net/nfs.cirrascale/allennlp/petew/cache:/root/.cache \
#--node_rank=$BEAKER_REPLICA_RANK
#  --nfs \
