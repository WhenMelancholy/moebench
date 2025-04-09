#!/bin/bash
#SBATCH --account=kempner_mzitnik_lab -p kempner_h100
#SBATCH -c 64
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -t 71:59:00
#SBATCH --mem=800G
#SBATCH -o logs/convert/%A_%a_%j.out
#SBATCH -e logs/convert/%A_%a_%j.err
#SBATCH --mail-user=mufan@cs.unc.edu
#SBATCH --mail-type=FAIL

set -e

cd /n/home08/zkong/mufan/tmp/moebench/OLMo
folders=$(ls runs/)
for folder in $folders; do
    if [ -d "runs/$folder/latest" ]; then
        echo "Processing $folder"
        full_relative_path=runs/$folder/latest
        # Check if $full_relative_path/config.json exists, if exists, skip
        if [ -f "$full_relative_path/config.json" ]; then
            echo "Skipping $folder, config.json already exists"
            continue
        fi

        python hf_olmo/convert_olmo_to_hf.py --checkpoint-dir $full_relative_path --tokenizer allenai/gpt-neox-olmo-dolma-v1_5 --keep-olmo-artifacts
    fi
done
