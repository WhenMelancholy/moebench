#!/bin/bash
#SBATCH --job-name=olmoe_training           # 作业名称
#SBATCH --nodes=32                          # 使用的节点数
#SBATCH --ntasks-per-node=8                 # 每个节点上的任务数（即每个节点的 GPU 数）
#SBATCH --cpus-per-task=8                   # 每个任务分配的 CPU 核心数
#SBATCH --gres=gpu:8                        # 每个节点分配的 GPU 数
#SBATCH --partition=your_partition          # 指定分区名称
#SBATCH --output=%x_%j.out                  # 标准输出文件
#SBATCH --error=%x_%j.err                   # 标准错误文件
#SBATCH --time=72:00:00                     # 作业运行时间上限

set -ex

# 配置文件路径
CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-final.yml

# 训练参数
ARGS='--run_name=olmoe-8x1b-newhp-newds-final --save-overwrite --fsdp.sharding_strategy=FULL_SHARD --device_train_microbatch_size=4 --canceled_check_interval=9999999'

# 环境变量设置
export HF_DATASETS_OFFLINE=1
export NCCL_IB_HCA=^=mlx5_bond_0
export OMP_NUM_THREADS=8
export OLMO_TASK=model

# 设置主节点地址和端口
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29400

# 计算节点总数和当前节点排名
NNODES=$SLURM_JOB_NUM_NODES
NODE_RANK=$SLURM_NODEID

# 加载必要的模块或激活虚拟环境
# module load your_module
# source activate your_virtual_env

# 安装或升级所需的 Python 包
pip install --upgrade torch==2.3.0
pip install --upgrade flash-attn --no-build-isolation
pip install git+https://github.com/Muennighoff/megablocks.git@zloss

# 准备 Hugging Face 数据缓存
mkdir -p /root/.cache
pushd /root/.cache
curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -
popd

# 使用 torchrun 启动分布式训练
srun torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py $CONFIG_PATH $ARGS
