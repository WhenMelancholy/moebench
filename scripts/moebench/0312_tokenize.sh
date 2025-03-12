#!/bin/bash

set -e

ROOTDIR=/n/home08/zkong/mufan/tmp/moebench/

export NUM_PROCESSES=48
export PATH_TO_DOWNLOADED_DATA=${ROOTDIR}/OLMo/data/minipile-jsonl
export PATH_WHERE_TO_SAVE_TOKENIZED_DATA=${ROOTDIR}/OLMo/data/minipile-tokenized
splits=(train validation test)
for split in ${splits[@]}; do
    mkdir -p ${PATH_WHERE_TO_SAVE_TOKENIZED_DATA}/${split}
    dolma tokens \
        --documents ${PATH_TO_DOWNLOADED_DATA}/${split} \
        --destination ${PATH_WHERE_TO_SAVE_TOKENIZED_DATA}/${split} \
        --tokenizer.name_or_path 'allenai/gpt-neox-olmo-dolma-v1_5' \
        --max_size '2_147_483_648' \
        --seed 0 \
        --tokenizer.eos_token_id 50279 \
        --tokenizer.pad_token_id 1 \
        --processes ${NUM_PROCESSES}
done
