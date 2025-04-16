#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 64
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH -t 71:59:00
#SBATCH --mem=800G
#SBATCH -o logs/slurm_pretrain_auxfree/%A_%a_%j.out
#SBATCH -e logs/slurm_pretrain_auxfree/%A_%a_%j.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=FAIL
#SBATCH --array=6%4

set -ex

[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=0
cd /n/home08/zkong/mufan/tmp/moebench/OLMo/
experts=(8 16 8 16 32 64 32 64)
topks=(2 4 1 3 8 16 7 15)
mlp_ratios=(8 4 8 4 2 1 2 1)
shared_experts=(False False True True False False True True)
expert=${experts[$SLURM_ARRAY_TASK_ID]}
topk=${topks[$SLURM_ARRAY_TASK_ID]}
mlp_ratio=${mlp_ratios[$SLURM_ARRAY_TASK_ID]}
shared_expert=${shared_experts[$SLURM_ARRAY_TASK_ID]}

export CONFIG_NAME=0312-OLMoE-300M-15ep
CONFIG_PATH=configs/${CONFIG_NAME}.yml
ARGS="--run_name=${CONFIG_NAME}-${topk}of${expert}-shared${shared_expert}-auxfree --save-overwrite --fsdp.sharding_strategy=FULL_SHARD --device_train_microbatch_size=8 --canceled_check_interval=9999999 --model.moe_num_experts=${expert} --model.moe_top_k=${topk} --model.mlp_ratio=${mlp_ratio} --model.moe_shared_expert=${shared_expert} --model.bias_u=0.001 --model.bias_update_step=64"
DATE=0319

# export WANDB_API_KEY="dc819d760ed4bb33f5565fd184c37dd03b5b35e4"
export WANDB_PROJECT=olmoe
export WANDB_NAME="${DATE}-${CONFIG_NAME}-${topk}of${expert}"
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
