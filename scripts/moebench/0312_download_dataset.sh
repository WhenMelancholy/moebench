#!/bin/bash

set -e

cd /n/home08/zkong/mufan/tmp/moebench/

# HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type dataset --local-dir OLMo/data/OLMoE-mix-0924 allenai/OLMoE-mix-0924
# Potential dataset: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample?row=0, though 1T may also be too large for us
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type dataset --local-dir OLMo/data/minipile JeanKaddour/minipile
