#!/bin/bash

set -e

# TODO: 清理掉已经被转换成 hf checkpoint 的 pytorch_model 文件以便节省空间
cd /n/home08/zkong/mufan/tmp/moebench/
checkpoint_dirs=open-instruct/output
for checkpoint_dir in $(ls $checkpoint_dirs/); do
    # 检查是否存在 pytorch_model.bin
    # echo "Processing checkpoint directory: $checkpoint_dir"
    for epoch_dir in $(ls $checkpoint_dirs/$checkpoint_dir/ | grep epoch); do
        epoch_dir="$checkpoint_dirs/$checkpoint_dir/$epoch_dir"
        echo "Processing epoch directory: $epoch_dir"
        # 检查是否存在 pytorch_model.bin
        if [[ -d "$epoch_dir/pytorch_model" ]]; then
            if [[ -f "$epoch_dir/config.json" ]]; then
                # 如果存在 config.json，说明已经转换成了 HF checkpoint
                # 删除 pytorch_model.bin 以节省空间
                echo "Deleting pytorch_model.bin in $epoch_dir"
                rm -rf "$epoch_dir/pytorch_model"
            else
                echo "No config.json found in $epoch_dir, skipping deletion."
            fi
        fi
    done
done
