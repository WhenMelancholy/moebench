#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 64
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH -t 1-00:05
#SBATCH --mem=800G
#SBATCH -o logs/slurm_benchmark_deepseek/%j_%A_%a.out
#SBATCH -e logs/slurm_benchmark_deepseek/%j_%A_%a.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=FAIL
set -euo pipefail

array_ids=(0 1 2 9 12 14)

job_count=0
parallel_jobs=4
for index in "${array_ids[@]}"; do
    export SLURM_ARRAY_TASK_ID=$index
    export CUDA_VISIBLE_DEVICES=$job_count
    job_count=$((job_count + 1))

    echo "Running job with SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID and CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    bash 0423_benchmark_setting_ds.sh &
    if [ $job_count -ge $parallel_jobs ]; then
        wait
        job_count=0
    fi
done
