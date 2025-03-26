#!/bin/bash
set -e

cd /n/home08/zkong/mufan/tmp/moebench/OLMo
folders=$(ls runs/)
for folder in $folders; do
    if [ -d "runs/$folder/latest" ]; then
        echo "Processing $folder"
        full_relative_path=runs/$folder/latest
        python hf_olmo/convert_olmo_to_hf.py --checkpoint-dir $full_relative_path --tokenizer allenai/gpt-neox-olmo-dolma-v1_5 --keep-olmo-artifacts
    fi
done
